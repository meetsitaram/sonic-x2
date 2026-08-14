#!/usr/bin/env python3
"""ONNX-driven MuJoCo evaluation for X2 Ultra.

Point it at the fused SONIC ONNX and a motion-lib PKL and it launches a
MuJoCo viewer with the policy tracking the clip; ``--no-viewer`` runs the
same loop headless, ``--kinematic`` plays the reference without physics,
``--record`` captures either mode to mp4.

The fused ONNX takes a single 1670-D vector::

    actor_obs = [tokenizer_obs(680) | proprioception(990)]

and returns a 31-D action in IsaacLab DOF order. Observation construction
(and all X2 constants) come from :mod:`eval_x2_mujoco`, shared with the
deployment stack. The real robot's deploy tuning preset (PD trim, target
clamps, LPF, action clip) is applied by default — see ``--tuning``.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

# Reuse all constants and helpers from eval_x2_mujoco.py — by importing rather
# than copy-pasting, both scripts stay in lockstep if the X2 constants ever
# change (kp/kd/action_scale/joint maps/default angles all derive from the
# same source of truth).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_x2_mujoco import (  # noqa: E402  (sys.path setup must come first)
    ACTION_SCALE,
    CONTROL_DT,
    DECIMATION,
    DEFAULT_DOF,
    IL_TO_MJ_DOF,
    JOINT_TO_ACTUATOR,
    KD,
    KP,
    MJ_TO_IL_DOF,
    MJCF_PATH,
    NUM_DOFS,
    SIM_DT,
    ProprioceptionBuffer,
    build_tokenizer_obs,
    compute_motion_state,
    get_motion_fps,
    get_total_frames,
    load_deploy_tuning,
    load_motion_data,
    quat_rotate_inverse,
)


# Tokenizer layout constants for the X2 g1 encoder (sourced from training
# config gear_sonic/config/exp/manager/universal_token/all_modes/sonic_x2_ultra*):
#   command_multi_future_nonflat:    (NUM_FUTURE_FRAMES=10, COMMAND_DIM_PER_FRAME=62)
#   motion_anchor_ori_b_mf_nonflat:  (NUM_FUTURE_FRAMES=10, ORI_DIM_PER_FRAME=6)
# Total tokenizer width: 10*62 + 10*6 = 680.
NUM_FUTURE_FRAMES_TOK = 10
COMMAND_DIM_PER_FRAME = 62
ORI_DIM_PER_FRAME = 6
COMMAND_FLAT_DIM = NUM_FUTURE_FRAMES_TOK * COMMAND_DIM_PER_FRAME  # 620
ORI_FLAT_DIM = NUM_FUTURE_FRAMES_TOK * ORI_DIM_PER_FRAME  # 60
TOK_DIM = COMMAND_FLAT_DIM + ORI_FLAT_DIM  # 680
PROP_DIM = 990
ACTOR_OBS_DIM = TOK_DIM + PROP_DIM  # 1670


# NOTE on the tokenizer layout the fused g1 ONNX expects (verified
# 2026-05-01 by static analysis of the exported graph + parity test
# against a fresh ``dump_isaaclab_step0`` dump):
#
#     The first 680 elements of ``obs`` are reshaped DIRECTLY to
#     (B, 10, 68) by the ONNX graph (single ``Reshape(-1, 10, 68)`` op
#     after a ``Slice``), then flattened back to (B, 680) for the
#     encoder MLP. This means the ONNX expects per-frame *interleaved*
#     layout::
#
#         [cmd_f0(62) | ori_f0(6) | cmd_f1(62) | ori_f1(6) | ... | cmd_f9(62) | ori_f9(6)]
#
#     i.e. exactly what ``np.concatenate([cmd(10,62), ori(10,6)],
#     axis=-1).reshape(-1)`` produces — which is precisely the layout
#     ``eval_x2_mujoco.build_tokenizer_obs`` (and the live IsaacLab
#     ``encoder_input_full``) emits. NO REARRANGEMENT is needed at the
#     ONNX boundary.
#
# History (kept for posterity): an earlier version of this file had a
# ``_interleaved_to_grouped`` rearrangement based on a misreading of
# ``UniversalTokenWrapper.forward()``. That added rearrangement was
# the entire source of the "PT vs ONNX 3.3 rad delta" parity failures
# (e.g. neutral_walk init=20 falling at 1.64 s under ONNX while PT
# saturated). Removing the rearrangement makes ONNX agree with the
# live module to ~5e-7 rad on identical inputs.


# ---------- Offscreen video recorder ----------
class VideoRecorder:
    """Offscreen MuJoCo render piped to ffmpeg (H.264 mp4).

    Same tracking-camera framing as the interactive viewer. Needs ``ffmpeg``
    on PATH; headless GL (set ``MUJOCO_GL=egl`` if there is no display).
    Recording is optional — nothing else depends on it.
    """

    def __init__(self, path: str, model, track_body_id: int, fps: float,
                 width: int = 960, height: int = 720):
        import shutil
        import subprocess
        if shutil.which("ffmpeg") is None:
            raise SystemExit("--record needs ffmpeg on PATH")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = track_body_id
        self.cam.azimuth, self.cam.elevation, self.cam.distance = 120, -20, 2.5
        self.frames = 0
        self.path = path
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
             "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
             path],
            stdin=subprocess.PIPE,
        )

    def capture(self, data) -> None:
        self.renderer.update_scene(data, camera=self.cam)
        self.proc.stdin.write(self.renderer.render().tobytes())
        self.frames += 1

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()
        print(f"  [record] wrote {self.frames} frames -> {self.path}", flush=True)


def run_kinematic(args) -> None:
    """Kinematic reference playback (no physics, no policy): pose the robot
    straight from the clip frames (PKL ``dof`` is MJCF qpos convention).
    Viewer by default; with ``--record`` renders one offscreen pass over the
    selected clip(s) to mp4 at the clip fps instead."""
    import joblib
    data = joblib.load(args.motion)
    names = list(data.keys())
    if args.clip:
        if args.clip in names:
            names = [args.clip]
        else:
            names = [k for k in names if args.clip.lower() in k.lower()]
        if not names:
            raise SystemExit(f"--clip '{args.clip}' matched no clips in {args.motion}")
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    mjd = mujoco.MjData(model)
    pelvis = model.body("pelvis").id

    def pose(m, f):
        mjd.qpos[0:3] = m["root_trans_offset"][f]
        q = np.asarray(m["root_rot"][f])  # xyzw
        mjd.qpos[3:7] = [q[3], q[0], q[1], q[2]]
        mjd.qpos[7:7 + NUM_DOFS] = np.asarray(m["dof"][f])
        mjd.qvel[:] = 0
        mujoco.mj_forward(model, mjd)

    if args.record:
        rec = VideoRecorder(args.record, model, pelvis,
                            float(data[names[0]]["fps"]))
        for name in names:
            m = data[name]
            n_frames = np.asarray(m["dof"]).shape[0]
            print(f"[kinematic] {name}: {n_frames} frames @ {m['fps']:g} fps",
                  flush=True)
            for f in range(args.init_frame, n_frames):
                pose(m, f)
                rec.capture(mjd)
        rec.close()
        return

    state = {"paused": False, "clip": 0, "frame": float(args.init_frame)}

    def key_cb(keycode):
        import glfw
        if keycode == glfw.KEY_SPACE:
            state["paused"] = not state["paused"]
        elif keycode == glfw.KEY_R:
            state["frame"] = float(args.init_frame)
        elif keycode == glfw.KEY_N:
            state["clip"] = (state["clip"] + 1) % len(names)
            state["frame"] = float(args.init_frame)

    print("Kinematic playback: SPACE pause, R restart, N next clip.", flush=True)
    with mujoco.viewer.launch_passive(
        model, mjd, key_callback=key_cb,
        show_left_ui=False, show_right_ui=False,
    ) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = pelvis
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 120, -20, 2.5
        while viewer.is_running():
            if state["paused"]:
                viewer.sync()
                time.sleep(0.02)
                continue
            m = data[names[state["clip"]]]
            n_frames = np.asarray(m["dof"]).shape[0]
            pose(m, int(state["frame"]) % n_frames)
            viewer.sync()
            time.sleep(1.0 / (float(m["fps"]) * max(args.speed, 1e-6)))
            state["frame"] += 1


# ---------- ONNX wrapper ----------
class OnnxActor:
    """Thin wrapper that mimics ``UniversalTokenActor.__call__`` signature.

    Accepts ``proprioception (990)`` and ``tokenizer_obs (680)`` numpy arrays
    (single batch). ``tokenizer_obs`` must be in the per-frame interleaved
    layout produced by :func:`eval_x2_mujoco.build_tokenizer_obs`; the ONNX
    graph consumes it directly with no rearrangement. See the module-level
    note above for the layout rationale.
    """

    def __init__(self, onnx_path: str, providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        # The fused actor is a small MLP: more threads = pure spin-wait
        # overhead. Uncapped, ORT grabs every core (~10 cores busy-spinning
        # per viewer) which starves the GL render loop and drops the sim
        # below real time on laptops. 2 threads is already memory-bound.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("ORT_NUM_THREADS", "2"))
        opts.inter_op_num_threads = 1
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            onnx_path, sess_options=opts, providers=providers
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                f"Expected exactly 1 input and 1 output on the fused ONNX, "
                f"got {len(inputs)} inputs / {len(outputs)} outputs"
            )
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.input_shape = inputs[0].shape
        self.output_shape = outputs[0].shape
        actual_width = self.input_shape[-1]
        if actual_width != ACTOR_OBS_DIM:
            raise RuntimeError(
                f"ONNX input width {actual_width} != expected {ACTOR_OBS_DIM} "
                f"({TOK_DIM} tokenizer + {PROP_DIM} proprioception). Was this ONNX "
                f"exported from a different model than X2 Ultra g1+g1_dyn?"
            )

    def __call__(self, proprioception: np.ndarray, tokenizer_obs: np.ndarray) -> np.ndarray:
        if tokenizer_obs.shape[-1] != TOK_DIM:
            raise ValueError(
                f"Expected tokenizer width {TOK_DIM}, got {tokenizer_obs.shape[-1]}"
            )
        actor_obs = np.concatenate(
            [tokenizer_obs.astype(np.float32), proprioception.astype(np.float32)]
        ).reshape(1, -1)
        out = self.session.run([self.output_name], {self.input_name: actor_obs})[0]
        return out[0]  # (31,) IL order

    def describe(self) -> str:
        return (
            f"input '{self.input_name}' shape={self.input_shape} -> "
            f"output '{self.output_name}' shape={self.output_shape}"
        )



# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--onnx",
        default=None,
        help=(
            "Path to fused encoder+decoder ONNX (e.g. model_step_002000_g1.onnx). "
            "When omitted, falls back to the SONIC model cache: "
            "$SONIC_X2_MODELS/sonic_policy/x2_sonic_policy.onnx, else "
            "$SONIC_HOME/x2/... (SONIC_HOME defaults to ~/.cache/sonic; "
            "populate via install_scripts/setup_x2.sh). Required if no "
            "cached model exists."
        ),
    )
    parser.add_argument("--motion", required=True, help="Reference motion PKL.")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--tuning",
        default="__default__",
        help="Real-deploy tuning preset YAML "
        "(gear_sonic_deploy/configs/real_deploy_tuning/*.yaml). Applies the "
        "robot's per-group PD trim (replacing the sim-only ankle bump), "
        "target-deviation clamps, target LPF, and action clip so the sim "
        "matches the real deployment. Default: walk_101.yaml — the preset "
        "the robot is deployed with. Pass --tuning '' for raw "
        "training-parity gains (parity/eval baselines).",
    )
    parser.add_argument(
        "--init-frame",
        type=int,
        default=0,
        help="Motion frame to RSI-initialize the robot at (default 0).",
    )
    parser.add_argument(
        "--fall-height",
        type=float,
        default=0.4,
        help="Pelvis z below this (m) triggers a reset (default 0.4).",
    )
    parser.add_argument(
        "--fall-tilt-cos",
        type=float,
        default=-0.3,
        help="gravity_body[z] above this triggers a reset (default -0.3 ~ 72 deg tilt).",
    )
    parser.add_argument(
        "--max-episode",
        type=float,
        default=0.0,
        help="If > 0, force-reset after this many simulated seconds *per episode* "
        "(default 0 = no per-episode limit).",
    )
    parser.add_argument(
        "--total-sim-seconds",
        type=float,
        default=0.0,
        help="If > 0 (and --no-viewer), exit once *cumulative* simulated seconds "
        "across all episodes reach this. Use this for a fixed-budget parity "
        "rollout that auto-resets through falls (default 0 = no cumulative cap).",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Headless mode: no MuJoCo viewer, no real-time pacing. Pair with "
        "--total-sim-seconds for a deterministic CI-style parity check.",
    )
    parser.add_argument(
        "--record",
        default=None,
        metavar="OUT.mp4",
        help="Record an offscreen 25 fps video of the rollout (headless "
             "only — combine with --no-viewer). Needs ffmpeg on PATH. "
             "With --kinematic, records at the clip fps instead.",
    )
    parser.add_argument(
        "--kinematic",
        action="store_true",
        help="Kinematic reference playback: pose the robot directly from "
             "the clip frames (no physics, no policy, no ONNX needed). "
             "Viewer, or offscreen mp4 with --record.",
    )
    parser.add_argument(
        "--clip",
        default=None,
        help="Kinematic mode: exact clip key, or substring filter "
             "(default: all clips in the PKL).",
    )
    args = parser.parse_args()

    if args.kinematic:
        if not args.motion:
            parser.error("--kinematic requires --motion")
        run_kinematic(args)
        return

    if args.onnx is None:
        # Default resolution order (mirrors the stack scripts): explicit
        # --onnx > $SONIC_X2_MODELS > $SONIC_HOME/x2 (~/.cache/sonic/x2).
        cache_root = os.environ.get("SONIC_X2_MODELS") or os.path.join(
            os.environ.get("SONIC_HOME", os.path.expanduser("~/.cache/sonic")),
            "x2",
        )
        candidate = os.path.join(cache_root, "sonic_policy", "x2_sonic_policy.onnx")
        if os.path.isfile(candidate):
            print(f"--onnx omitted; using SONIC model cache: {candidate}", flush=True)
            args.onnx = candidate
        else:
            parser.error(
                "--onnx is required (no cached model at "
                f"{candidate}; run install_scripts/setup_x2.sh or pass --onnx)."
            )

    print(f"Loading ONNX session from {args.onnx} ...", flush=True)
    onnx_actor = OnnxActor(args.onnx)
    print(f"  ONNX: {onnx_actor.describe()}", flush=True)

    # Real-deploy tuning preset: swaps in the robot's effective PD gains and
    # reproduces the deploy safety stack (target clamp + LPF + action clip).
    tuning = None
    kp_run, kd_run = KP, KD
    _lpf_y = {"y": None}  # target LPF state; reset with each episode
    if args.tuning == "__default__":
        # walk_101 is the preset the reference robot deploys with; default
        # to it so sim behavior matches the real robot out of the box.
        default_yaml = (
            Path(__file__).resolve().parents[1]
            / "configs/real_deploy_tuning/walk_101.yaml"
        )
        args.tuning = str(default_yaml) if default_yaml.is_file() else ""
    if args.tuning:
        tuning = load_deploy_tuning(args.tuning)
        kp_run, kd_run = tuning["kp"], tuning["kd"]
        print(
            f"  [tuning] {args.tuning}: PD trim + target clamps + LPF + "
            f"action_clip={tuning['action_clip']:g} (robot-matched)",
            flush=True,
        )

    print(f"Loading motion from {args.motion} ...", flush=True)
    motion_data = load_motion_data(args.motion)
    total_frames = get_total_frames(motion_data)
    motion_fps = get_motion_fps(motion_data)
    print(
        f"  {total_frames} frames @ {motion_fps} fps = {total_frames / motion_fps:.1f}s",
        flush=True,
    )

    print("Loading MuJoCo model ...", flush=True)
    mj_model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = SIM_DT

    pelvis_id = mj_model.body("pelvis").id

    recorder = None
    if args.record:
        if not args.no_viewer:
            parser.error("--record requires --no-viewer (offscreen render)")
        recorder = VideoRecorder(args.record, mj_model, pelvis_id, 25.0)
        print(f"Recording -> {args.record} @ 25 fps", flush=True)

    init_frame = int(args.init_frame)
    init_motion_state = compute_motion_state(motion_data, init_frame, motion_fps)
    init_root_z = float(init_motion_state["root_pos_w"][2])
    print(
        f"  [RSI] Initializing from motion frame {init_frame} "
        f"(t={init_frame / motion_fps:.3f}s)",
        flush=True,
    )

    prop_buf = ProprioceptionBuffer()
    last_action_mj = np.zeros(NUM_DOFS, dtype=np.float32)
    sim_time = float(init_frame) / motion_fps
    step_count = 0
    episode_count = 0
    episode_start_step = 0
    paused = False

    def _apply_init_state():
        _lpf_y["y"] = None
        s = init_motion_state
        mj_data.qpos[0] = 0.0
        mj_data.qpos[1] = 0.0
        mj_data.qpos[2] = float(s["root_pos_w"][2])
        mj_data.qpos[3:7] = s["root_quat_w_wxyz"]
        mj_data.qpos[7 : 7 + NUM_DOFS] = s["joint_pos_mj"]
        mj_data.qvel[0:3] = s["root_lin_vel_w"]
        mj_data.qvel[3:6] = quat_rotate_inverse(
            s["root_quat_w_wxyz"], s["root_ang_vel_w"]
        )
        mj_data.qvel[6 : 6 + NUM_DOFS] = s["joint_vel_mj"]
        mj_data.xfrc_applied[:] = 0
        mujoco.mj_forward(mj_model, mj_data)

    _apply_init_state()

    # Tracks the reference index so we can catch the loop wrap. Sonic is a tracking
    # policy: it can neither hold a frozen frame nor chase a teleport. Letting the
    # reference index wrap on its own snaps it back to frame 0 -- metres behind the
    # robot on locomotion clips -- and the robot collapses. Wrapping must be a full
    # reset so sim state and reference move together.
    prev_motion_frame = -1

    def reset_state(reason: str = "") -> None:
        nonlocal sim_time, last_action_mj, episode_count, episode_start_step
        nonlocal prev_motion_frame
        prev_motion_frame = -1
        sim_time = float(init_frame) / motion_fps
        last_action_mj[:] = 0
        prop_buf.reset()
        _apply_init_state()
        episode_count += 1
        episode_start_step = step_count
        tag = f" ({reason})" if reason else ""
        print(f"\n[reset]{tag} starting episode {episode_count}", flush=True)

    # Per-step body that's identical in headless and viewer paths.
    def step_once() -> str | None:
        """Run one control tick. Returns reset reason if the episode ended."""
        nonlocal sim_time, step_count, last_action_mj, prev_motion_frame

        motion_time = sim_time * args.speed
        motion_frame = int(motion_time * motion_fps) % total_frames
        # Clip looped: end the episode instead of teleporting the reference (see above).
        if 0 <= prev_motion_frame and motion_frame < prev_motion_frame:
            return "motion_end"
        prev_motion_frame = motion_frame
        motion_time = motion_frame / motion_fps

        qpos_j = mj_data.qpos[7 : 7 + NUM_DOFS].copy()
        qvel_j = mj_data.qvel[6 : 6 + NUM_DOFS].copy()
        base_quat = mj_data.qpos[3:7].copy()
        base_angvel = mj_data.qvel[3:6].copy()

        dof_pos_il = qpos_j[IL_TO_MJ_DOF]
        dof_vel_il = qvel_j[IL_TO_MJ_DOF]
        action_il = last_action_mj[IL_TO_MJ_DOF]

        gravity = quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0]))
        dof_pos_rel_il = dof_pos_il - DEFAULT_DOF[IL_TO_MJ_DOF]

        prop_buf.append(gravity, base_angvel, dof_pos_rel_il, dof_vel_il, action_il)
        proprioception = prop_buf.get_flat()
        tokenizer_obs = build_tokenizer_obs(motion_data, motion_time, base_quat, motion_fps)

        # ONNX action (always computed).
        action_il_onnx = onnx_actor(proprioception, tokenizer_obs)


        action_il_drive = action_il_onnx

        action_mj = action_il_drive[MJ_TO_IL_DOF]
        if tuning is not None:
            action_mj = np.clip(
                action_mj, -tuning["action_clip"], tuning["action_clip"]
            )
        last_action_mj = action_mj.astype(np.float32).copy()
        target_pos = DEFAULT_DOF + action_mj * ACTION_SCALE
        if tuning is not None:
            # Deploy safety stack: clamp the target around the default pose,
            # then low-pass filter it (both per joint group).
            target_pos = np.clip(
                target_pos,
                DEFAULT_DOF - tuning["max_target_dev"],
                DEFAULT_DOF + tuning["max_target_dev"],
            )
            if _lpf_y["y"] is None:
                _lpf_y["y"] = target_pos.copy()
            else:
                _lpf_y["y"] += tuning["lpf_alpha"] * (target_pos - _lpf_y["y"])
            target_pos = _lpf_y["y"].copy()

        for _ in range(DECIMATION):
            torque = (
                kp_run * (target_pos - mj_data.qpos[7 : 7 + NUM_DOFS])
                - kd_run * mj_data.qvel[6 : 6 + NUM_DOFS]
            )
            for j in range(NUM_DOFS):
                mj_data.ctrl[JOINT_TO_ACTUATOR[j]] = torque[j]
            mujoco.mj_step(mj_model, mj_data)

        sim_time += CONTROL_DT
        step_count += 1

        pelvis_z = float(mj_data.qpos[2])
        grav_z = float(gravity[2])
        episode_seconds = (step_count - episode_start_step) * CONTROL_DT
        if pelvis_z < args.fall_height:
            return f"pelvis_z={pelvis_z:.3f} < {args.fall_height:.2f}"
        if grav_z > args.fall_tilt_cos:
            tilt_deg = int(np.rad2deg(np.arccos(np.clip(-grav_z, -1, 1))))
            return (
                f"gravity_body[z]={grav_z:+.2f} > {args.fall_tilt_cos:.2f} "
                f"(tilt {tilt_deg} deg)"
            )
        if args.max_episode > 0 and episode_seconds >= args.max_episode:
            return f"reached --max-episode={args.max_episode:.1f}s"
        if step_count % 250 == 0:
            extra = ""
            if action_il_pt is not None:
                extra = (
                    f"  delta_inf={float(np.max(np.abs(action_il_pt - action_il_onnx))):.2e}"
                )
            print(
                f"[ep {episode_count}] step={step_count} sim={sim_time:.2f}s "
                f"frame={motion_frame}/{total_frames} h={pelvis_z:.3f}m{extra}",
                flush=True,
            )
        return None

    print("\n=== X2 MuJoCo Eval (ONNX) ===", flush=True)
    print(f"Robot RSI-initialized from motion frame {init_frame}.", flush=True)
    print(
        f"Auto-reset triggers: pelvis_z < {args.fall_height:.2f} m, "
        f"or gravity_body[z] > {args.fall_tilt_cos:.2f}.",
        flush=True,
    )
    if args.max_episode > 0:
        print(f"Max episode length: {args.max_episode:.1f} s.", flush=True)
    if not args.no_viewer:
        print("Press SPACE pause, R reset, V toggle camera.\n", flush=True)
    else:
        print("Headless mode (no viewer).\n", flush=True)

    # Headless exit semantics:
    #   --total-sim-seconds > 0  -> keep cycling resets until cumulative sim
    #                                time hits the cap (good for parity budgets)
    #   --max-episode > 0 only   -> exit after the first episode terminates
    #   neither set              -> run forever (Ctrl-C to stop)
    headless_exit_after_one_episode = (
        args.no_viewer and args.max_episode > 0 and args.total_sim_seconds <= 0
    )
    exit_requested = False
    cumulative_sim_seconds = 0.0

    if args.no_viewer:
        # Tight loop with no real-time pacing or viewer sync.
        while not exit_requested:
            reason = step_once()
            # 50 Hz control -> capture every 2nd tick = 25 fps video.
            if recorder is not None and reason is None and step_count % 2 == 0:
                recorder.capture(mj_data)
            cumulative_sim_seconds += CONTROL_DT
            if (
                args.total_sim_seconds > 0
                and cumulative_sim_seconds >= args.total_sim_seconds
            ):
                print(
                    f"  [end] cumulative sim time {cumulative_sim_seconds:.2f}s "
                    f">= --total-sim-seconds={args.total_sim_seconds:.1f}s, exiting.",
                    flush=True,
                )
                exit_requested = True
                continue
            if reason is not None:
                print(
                    f"  [end] ep={episode_count} ran "
                    f"{(step_count - episode_start_step) * CONTROL_DT:.2f}s, reason: {reason}",
                    flush=True,
                )
                if headless_exit_after_one_episode:
                    exit_requested = True
                else:
                    reset_state(reason)
        if recorder is not None:
            recorder.close()
    else:

        def key_callback(keycode):
            nonlocal paused
            import glfw

            if keycode == glfw.KEY_SPACE:
                paused = not paused
                print("Paused" if paused else "Resumed", flush=True)
            elif keycode == glfw.KEY_R:
                reset_state("manual")
            elif keycode == glfw.KEY_V:
                if viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                else:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    viewer.cam.trackbodyid = pelvis_id

        with mujoco.viewer.launch_passive(
            mj_model,
            mj_data,
            key_callback=key_callback,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            viewer.cam.azimuth = 120
            viewer.cam.elevation = -20
            viewer.cam.distance = 3.0
            viewer.cam.lookat[:] = [0.0, 0.0, init_root_z]
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = pelvis_id

            wall_start = time.time() - sim_time

            # viewer.sync() blocks on the display's vsync (~17 ms at 60 Hz),
            # so syncing every 50 Hz control step caps the sim below real
            # time. Render every Nth step instead: physics stays 50 Hz,
            # display runs at 25 Hz (override with VIEWER_RENDER_STRIDE=1
            # on machines with fast/uncomposited GL).
            render_stride = max(1, int(os.environ.get("VIEWER_RENDER_STRIDE", "2")))

            while viewer.is_running():
                if paused:
                    viewer.sync()
                    time.sleep(0.02)
                    continue

                reason = step_once()
                if step_count % render_stride == 0:
                    viewer.sync()

                wall_elapsed = time.time() - wall_start
                if sim_time > wall_elapsed:
                    time.sleep(sim_time - wall_elapsed)
                elif wall_elapsed - sim_time > 0.5:
                    # Fell >0.5 s behind (slow GL, window drag, CPU spike):
                    # rebase so we resume real-time pacing instead of
                    # chasing the deficit in permanent fast-forward.
                    wall_start = time.time() - sim_time

                if reason is not None:
                    print(
                        f"  [reset] ep={episode_count} ran "
                        f"{(step_count - episode_start_step) * CONTROL_DT:.2f}s, "
                        f"reason: {reason}",
                        flush=True,
                    )
                    reset_state(reason)
                    wall_start = time.time() - sim_time

        print("Viewer closed.")


if __name__ == "__main__":
    main()
