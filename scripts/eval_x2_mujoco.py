#!/usr/bin/env python3
"""X2 Ultra shared library for the MuJoCo player (no runnable main).

Single source of truth for the X2 constants (joint orders, kp/kd, action
scale, default pose, MJCF path), observation construction
(``build_tokenizer_obs``, ``ProprioceptionBuffer``), RSI state computation,
and the real-deploy tuning loader — all shared with the deployment stack.
``eval_x2_mujoco_onnx.py`` imports from here; nothing in this module needs
torch.
"""

import argparse
import collections
import math
import os
import time
from pathlib import Path

import joblib
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as Rot

GEAR_SONIC_ROOT = Path(__file__).resolve().parent.parent  # repo root (sonic-x2 standalone)

# ---------- X2 Ultra constants ----------
NUM_DOFS = 31
HISTORY_LEN = 10
CONTROL_DT = 0.02
SIM_DT = 0.005
DECIMATION = 4
NUM_FUTURE_FRAMES = 10
DT_FUTURE_REF = 0.1

MJCF_PATH = str(GEAR_SONIC_ROOT / "assets/mjcf/x2_ultra.xml")

MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
    "head_yaw_joint", "head_pitch_joint",
]

JOINT_TO_ACTUATOR = [
    0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11,
    12, 13, 14,
    17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30,
    15, 16,
]

IL_TO_MJ_DOF = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 29, 15, 22, 4, 10,
    30, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]
MJ_TO_IL_DOF = [
    0, 3, 6, 9, 14, 19, 1, 4, 7, 10, 15, 20, 2, 5, 8, 12,
    17, 21, 23, 25, 27, 29, 13, 18, 22, 24, 26, 28, 30, 11, 16,
]

NATURAL_FREQ = 10 * 2.0 * math.pi
DAMPING_RATIO = 2.0

# Default joint positions from x2_ultra.py InitialStateCfg (training reset pose)
DEFAULT_JOINT_POS = {
    "hip_pitch": -0.312,
    "knee": 0.669,
    "ankle_pitch": -0.363,
    "elbow": -0.6,
    "left_shoulder_roll": 0.2,
    "left_shoulder_pitch": 0.2,
    "right_shoulder_roll": -0.2,
    "right_shoulder_pitch": 0.2,
}

# Effort limits from x2_ultra.py actuator config. These feed
# action_scale = 0.25 * effort / kp, so they MUST match the limits the
# checkpoint was trained with or the action->target mapping is distorted
# (a stale waist 48 vs trained 36 over-pitched the torso 1.33x -> constant
# forward-fall/recovery-stepping on standing clips).
# Values below = x2_ultra.py after the 2026-07-14 motor-datasheet fix
# (waist pitch/roll 36 physical peak, wrist pitch/roll 6 per PFP-41-50).
# Checkpoints trained BEFORE that fix need waist 48 / wrist 4.8 here.
EFFORT_LIMITS = {
    "hip_yaw": 120.0, "hip_roll": 120.0, "hip_pitch": 120.0, "knee": 120.0,
    "ankle_pitch": 36.0, "ankle_roll": 24.0,
    "waist_yaw": 120.0, "waist_pitch": 36.0, "waist_roll": 36.0,
    "shoulder_pitch": 36.0, "shoulder_roll": 36.0, "shoulder_yaw": 24.0, "elbow": 24.0,
    "wrist_yaw": 24.0, "wrist_pitch": 6.0, "wrist_roll": 6.0,
    "head_yaw": 2.6, "head_pitch": 0.6,
}

ARMATURES = {
    "hip": 0.025101925, "knee": 0.025101925,
    "waist_yaw": 0.010177520, "waist_pitch": 0.003609725, "waist_roll": 0.003609725,
    "ankle": 0.003609725,
    "shoulder": 0.003609725, "elbow": 0.003609725,
    "wrist_yaw": 0.003609725, "wrist_pitch": 0.00425, "wrist_roll": 0.00425,
    "head": 0.00425,
}


# ---- Deployment-side per-joint-group PD scaling.
#
# Background (G5 + G16/G16b in docs/source/user_guide/sim2sim_mujoco.md):
# IsaacLab integrates PD against the joint-space inertia + armature implicitly,
# so the same numerical KP behaves stiffer than the explicit `ctrl`-driven
# torque MuJoCo applies during deployment. The standard fix used by every
# legged-RL deployment pipeline (incl. SONIC's own G1 deployment in
# `gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml`,
# which ships separate `JOINT_KP` and `MOTOR_KP` arrays) is to keep the
# trained policy as-is and bump the DEPLOYED PD per joint group to recover
# the lost loop gain.
#
# These multipliers are applied to KP and KD only — `action_scale` continues
# to be derived from the unscaled (training-equivalent) KP so the policy's
# action-to-target-offset mapping matches what was learned. Net effect: the
# same policy output produces a stiffer correction torque on the bumped
# joints.
#
# Values picked from G16b fine-grain sweep on
# `run-20260420_083925/model_step_006000.pt`, n=50 standing motions:
#   ankle KP ×1.5         → +0.51 s mean survival, 38W/10L (best surgical)
#   global KP ×1.3        → +0.47 s (broadband, equivalent)
#   ankle KP ×2.0 (4k)    → +0.85 s on the less-trained 4k checkpoint
# Conservative default below is 1.5× ankle KP, KD untouched, on the
# observation that:
#   - well-trained policies need smaller bumps (4k wanted ×2, 6k wants ×1.5)
#   - the response above 2.0 is non-monotone (signs of ringing)
#   - per-group asymmetry beyond ankle (waist down, knee KD down, etc.) DID
#     NOT help X2 — only the ankle bump transferred.
# Override per-run from CLI with --kp-scale-* / --kd-scale-* flags in
# `benchmark_motions_mujoco.py` (the benchmark scales multiply on top).
DEPLOYMENT_KP_SCALE = {
    "hip": 1.0, "knee": 1.0,
    "ankle": 1.5,
    "waist": 1.0,
    "shoulder": 1.0, "elbow": 1.0,
    "wrist": 1.0,
    "head": 1.0,
}
DEPLOYMENT_KD_SCALE = {
    "hip": 1.0, "knee": 1.0,
    "ankle": 1.0,
    "waist": 1.0,
    "shoulder": 1.0, "elbow": 1.0,
    "wrist": 1.0,
    "head": 1.0,
}


def _deployment_pd_scale(jname: str, table: dict) -> float:
    """Return the per-joint deployment scale for `jname` against `table`.

    Match patterns are evaluated against the MuJoCo joint name with `left_`/
    `right_` and `_joint` stripped. Most-specific patterns first ("ankle"
    before "wrist") so foot joints don't accidentally hit a generic key.
    """
    short = jname.replace("left_", "").replace("right_", "").replace("_joint", "")
    for token in ("hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist", "head"):
        if token in short:
            return float(table[token])
    return 1.0


def _compute_gains_and_scales():
    kp = np.zeros(NUM_DOFS, dtype=np.float64)
    kd = np.zeros(NUM_DOFS, dtype=np.float64)
    action_scale = np.ones(NUM_DOFS, dtype=np.float64)
    default_pos = np.zeros(NUM_DOFS, dtype=np.float64)

    for i, jname in enumerate(MUJOCO_JOINT_NAMES):
        # Training-equivalent PD from armature (the IsaacLab implicit values).
        kp_train = 0.0
        kd_train = 0.0
        for key, arm in ARMATURES.items():
            if key in jname:
                kp_train = arm * NATURAL_FREQ**2
                kd_train = 2.0 * DAMPING_RATIO * arm * NATURAL_FREQ
                break

        # Deployment KP/KD = training PD × per-group scale (G16b).
        kp[i] = kp_train * _deployment_pd_scale(jname, DEPLOYMENT_KP_SCALE)
        kd[i] = kd_train * _deployment_pd_scale(jname, DEPLOYMENT_KD_SCALE)

        # Action scale = 0.25 * effort / training KP (NOT scaled). This
        # preserves the policy's learned [-1, 1] → joint-target-offset
        # mapping; the deployment-side stiffening shows up purely as a
        # higher torque per unit error, not as a rescaled command range.
        short = jname.replace("_joint", "").replace("left_", "").replace("right_", "")
        for ekey, effort in EFFORT_LIMITS.items():
            if ekey in jname.replace("_joint", ""):
                action_scale[i] = 0.25 * effort / kp_train
                break

        # Default positions
        for dkey, dval in DEFAULT_JOINT_POS.items():
            if dkey == "left_shoulder_roll" and jname == "left_shoulder_roll_joint":
                default_pos[i] = dval; break
            elif dkey == "left_shoulder_pitch" and jname == "left_shoulder_pitch_joint":
                default_pos[i] = dval; break
            elif dkey == "right_shoulder_roll" and jname == "right_shoulder_roll_joint":
                default_pos[i] = dval; break
            elif dkey == "right_shoulder_pitch" and jname == "right_shoulder_pitch_joint":
                default_pos[i] = dval; break
            elif dkey in jname.replace("_joint", "") and "shoulder" not in dkey:
                default_pos[i] = dval; break

    return kp, kd, action_scale, default_pos


KP, KD, ACTION_SCALE, DEFAULT_DOF = _compute_gains_and_scales()

# Action clip to match the IsaacLab wrapper (config action_clip_value=20.0).
# Applied to the policy output each step so target + last_action history obs
# stay bounded exactly as in training. Override via env for A/B testing.
ACTION_CLIP = float(os.environ.get("ACTION_CLIP", "20.0"))


# ---------- Real-deploy tuning presets ----------
# Loads a gear_sonic_deploy/configs/real_deploy_tuning/*.yaml preset and
# reproduces the robot's runtime behavior in sim: per-group PD trim on top of
# the TRAINING-equivalent gains (replacing the sim-only DEPLOYMENT_*_SCALE
# bumps — the two must not stack), plus the C++ safety stack's per-group
# target-deviation clamps and target low-pass filters. Groups mirror
# tuning_config_to_args.py / x2_deploy_onnx_ref.cpp.
_TUNING_DEV_GROUPS = {
    "leg": range(0, 12), "waist": range(12, 15),
    "arm": range(15, 29), "head": range(29, 31),
}


def _tuning_trim_group(jname):
    """YAML kp/kd-trim group for a MuJoCo joint name (deploy convention)."""
    short = jname.replace("left_", "").replace("right_", "").replace("_joint", "")
    if "ankle_pitch" in short:
        return "ankle_pitch"
    if "ankle_roll" in short:
        return "ankle_roll"
    if "knee" in short:
        return "knee"
    if "hip" in short:
        return "hip"
    if short == "waist_yaw":
        return "waist_yaw"
    if short.startswith("waist"):
        return "waist_pr"
    if "wrist" in short:
        return "wrist"
    if "shoulder" in short or "elbow" in short:
        return "arm"
    if "head" in short:
        return "head"
    return None


def load_deploy_tuning(path):
    """Parse a real-deploy tuning YAML into per-joint sim arrays.

    Returns a dict with:
      kp, kd            (NUM_DOFS,) effective gains = training PD x YAML trim
      max_target_dev    (NUM_DOFS,) clamp half-width around DEFAULT_DOF (rad)
      lpf_alpha         (NUM_DOFS,) per-control-step first-order LPF blend
                        factor for targets (1.0 = filter disabled)
      action_clip       float
    """
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    kp = np.zeros(NUM_DOFS, dtype=np.float64)
    kd = np.zeros(NUM_DOFS, dtype=np.float64)
    for i, jname in enumerate(MUJOCO_JOINT_NAMES):
        kp_train = kd_train = 0.0
        for key, arm in ARMATURES.items():
            if key in jname:
                kp_train = arm * NATURAL_FREQ**2
                kd_train = 2.0 * DAMPING_RATIO * arm * NATURAL_FREQ
                break
        g = _tuning_trim_group(jname)
        kp[i] = kp_train * float(cfg.get(f"kp_scale_{g}", 1.0) or 1.0)
        kd[i] = kd_train * float(cfg.get(f"kd_scale_{g}", 1.0) or 1.0)

    dev_global = float(cfg.get("max_target_dev", np.inf) or np.inf)
    max_dev = np.full(NUM_DOFS, dev_global, dtype=np.float64)
    lpf_global = float(cfg.get("target_lpf_hz", 0.0) or 0.0)
    lpf_hz = np.full(NUM_DOFS, lpf_global, dtype=np.float64)
    for gname, rng in _TUNING_DEV_GROUPS.items():
        d = cfg.get(f"max_target_dev_{gname}")
        if d is not None and float(d) >= 0:
            max_dev[list(rng)] = float(d)
        lz = cfg.get(f"target_lpf_hz_{gname}")
        if lz is not None:
            lpf_hz[list(rng)] = float(lz)

    # First-order LPF blend per CONTROL step: y += alpha * (x - y);
    # alpha = 1 - exp(-2*pi*fc*dt). fc <= 0 disables (alpha = 1).
    lpf_alpha = np.ones(NUM_DOFS, dtype=np.float64)
    active = lpf_hz > 0.0
    lpf_alpha[active] = 1.0 - np.exp(-2.0 * math.pi * lpf_hz[active] * CONTROL_DT)

    return {
        "kp": kp,
        "kd": kd,
        "max_target_dev": max_dev,
        "lpf_alpha": lpf_alpha,
        "action_clip": float(cfg.get("action_clip", ACTION_CLIP) or ACTION_CLIP),
    }

# --- Wrist diagnostic dump (gated by WRIST_DUMP=<csv path>). Records, per
# control step: wrist target_pos, actual qpos, applied-vs-clamped torque, and
# joint-limit / ctrl-range saturation. Separates a bad-target (mapping/sign)
# bug from a bad-tracking (torque/limit) bug. ---
_WRIST_MJ_IDX = [i for i, n in enumerate(MUJOCO_JOINT_NAMES) if "wrist" in n]
_WRIST_NAMES = [MUJOCO_JOINT_NAMES[i].replace("_joint", "") for i in _WRIST_MJ_IDX]
# IL-order indices of the wrist joints (action vector is in IL order).
_WRIST_IL_IDX = [il for il in range(NUM_DOFS) if IL_TO_MJ_DOF[il] in _WRIST_MJ_IDX]
_wrist_dump_fh = None


def _wrist_dump(step, target_pos, mj_data, action_mj=None):
    global _wrist_dump_fh
    path = os.environ.get("WRIST_DUMP")
    if not path:
        return
    q = mj_data.qpos[7:7 + NUM_DOFS]
    qd = mj_data.qvel[6:6 + NUM_DOFS]
    tau = KP * (target_pos - q) - KD * qd
    if _wrist_dump_fh is None:
        _wrist_dump_fh = open(path, "w")
        hdr = ["step", "act_max", "act_mean"]
        for n in _WRIST_NAMES:
            hdr += [f"{n}_tgt", f"{n}_q", f"{n}_tau"]
        _wrist_dump_fh.write(",".join(hdr) + "\n")
    amax = float(np.abs(action_mj).max()) if action_mj is not None else 0.0
    amean = float(np.abs(action_mj).mean()) if action_mj is not None else 0.0
    row = [str(step), f"{amax:.3f}", f"{amean:.3f}"]
    for i in _WRIST_MJ_IDX:
        row += [f"{target_pos[i]:.4f}", f"{q[i]:.4f}", f"{tau[i]:.3f}"]
    _wrist_dump_fh.write(",".join(row) + "\n")
    _wrist_dump_fh.flush()


# ---------- Motion helpers ----------

# Nominal standing pelvis height used to rebase robot-recorded clips.
# Clips captured on the real robot (gear_sonic/data/motions/
# x2_recorded_gestures/) store odometry z, which is zeroed at ignition —
# their root_trans_offset[:, 2] is ~0 for the whole clip. RSI-initializing
# from that spawns the robot inside the floor and trips the pelvis_z
# auto-reset forever. Retargeted clips carry real pelvis height and are
# left untouched (the 0.3 m threshold cleanly separates the two).
STAND_ROOT_Z = 0.60


def load_motion_data(path):
    data = joblib.load(path)
    for m in data.values():
        rt = m.get("root_trans_offset")
        if rt is None:
            continue
        rt = np.asarray(rt)
        if float(np.abs(rt[:, 2]).max()) < 0.3:
            rt = rt.astype(np.float64, copy=True)
            rt[:, 2] += STAND_ROOT_Z
            m["root_trans_offset"] = rt
    return data




def _m(data):
    return data[list(data.keys())[0]]


def get_total_frames(data):
    return _m(data)["dof"].shape[0]


def get_motion_fps(data):
    return float(_m(data)["fps"])


def compute_motion_state(motion_data, frame, fps):
    """Reconstruct the full robot reset state from a motion frame.

    Mirrors IsaacLab's Reference State Initialization (RSI) in
    ``TrackingCommand`` (see ``gear_sonic/envs/manager_env/mdp/commands.py``
    ``write_joint_state_to_sim`` / ``write_root_state_to_sim``): at reset,
    the simulator teleports the robot to the motion's frame ``f`` state —
    joint pos+vel, root pos+quat+lin_vel+ang_vel — bypassing physics. We
    reproduce that here so MuJoCo starts in the same distribution the
    policy was trained on.

    The PKL stores joint DOFs in **MuJoCo order**, ``root_trans_offset`` in
    world frame, and ``root_rot`` as a scipy-style **xyzw** quaternion.
    Velocities are not stored in the PKL, so we reconstruct them with a
    one-step forward finite difference (matches IsaacLab's motion-lib
    ``dof_vels`` and ``global_root_velocity`` to ~1e-2 precision).

    Returns a dict with:
        joint_pos_mj      (31,) MuJoCo joint order
        joint_vel_mj      (31,) MuJoCo joint order
        root_pos_w        (3,)  world-frame xyz
        root_quat_w_wxyz  (4,)  MuJoCo wxyz quaternion
        root_lin_vel_w    (3,)  world-frame linear vel
        root_ang_vel_w    (3,)  world-frame angular vel (axis-angle / dt)
    """
    m = _m(motion_data)
    n_frames = m["dof"].shape[0]
    f = int(frame) % n_frames
    f_next = min(f + 1, n_frames - 1)
    dt = 1.0 / float(fps)

    joint_pos_mj = np.asarray(m["dof"][f], dtype=np.float64)
    if f_next != f:
        joint_vel_mj = (np.asarray(m["dof"][f_next], dtype=np.float64) - joint_pos_mj) / dt
    else:
        joint_vel_mj = np.zeros(NUM_DOFS, dtype=np.float64)

    root_pos_w = np.asarray(m["root_trans_offset"][f], dtype=np.float64).copy()

    # PKL root_rot is xyzw (scipy). MuJoCo qpos[3:7] is wxyz.
    quat_xyzw = np.asarray(m["root_rot"][f], dtype=np.float64)
    root_quat_w_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64
    )

    if f_next != f:
        root_lin_vel_w = (
            np.asarray(m["root_trans_offset"][f_next], dtype=np.float64) - root_pos_w
        ) / dt
        q_next_xyzw = np.asarray(m["root_rot"][f_next], dtype=np.float64)
        rel = Rot.from_quat(q_next_xyzw) * Rot.from_quat(quat_xyzw).inv()
        root_ang_vel_w = rel.as_rotvec() / dt
    else:
        root_lin_vel_w = np.zeros(3, dtype=np.float64)
        root_ang_vel_w = np.zeros(3, dtype=np.float64)

    return {
        "joint_pos_mj": joint_pos_mj,
        "joint_vel_mj": joint_vel_mj,
        "root_pos_w": root_pos_w,
        "root_quat_w_wxyz": root_quat_w_wxyz,
        "root_lin_vel_w": root_lin_vel_w,
        "root_ang_vel_w": root_ang_vel_w,
    }


# ---------- Observation construction ----------
# IsaacLab observation layout for policy group (concatenate_terms=True).
# CRITICAL: term order follows the PolicyCfg dataclass ATTRIBUTE order
# (gear_sonic/envs/manager_env/mdp/observations.py lines 107-128), NOT the
# YAML ordering. The PolicyAtmCfg comment confirms it explicitly:
#   "Order matches PolicyCfg: base_ang_vel, joint_pos, joint_vel, actions, gravity_dir"
# Each term: history_length=10, oldest at index 0, newest at index max_length-1
# (CircularBuffer.buffer property; broadcast-fills all slots with first sample
# at reset). Layout (oldest..newest within each term):
#   base_ang_vel  (3)  x 10 -> 30
#   joint_pos_rel (31) x 10 -> 310
#   joint_vel     (31) x 10 -> 310
#   last_action   (31) x 10 -> 310
#   gravity_dir   (3)  x 10 -> 30
# Total proprioception: 990
# Verified via /tmp/x2_step0_isaaclab.pt dump (slicing each block matched the
# corresponding env_state quantity within the configured noise tolerance).

def quat_rotate_inverse(q_wxyz, v):
    """Rotate vector v by the INVERSE of quaternion q (wxyz convention).

    Matches IsaacLab's quat_apply_inverse: v - w*t + cross(u, t).
    """
    w, x, y, z = q_wxyz
    u = np.array([x, y, z])
    t = 2.0 * np.cross(u, v)
    return v - w * t + np.cross(u, t)


class ProprioceptionBuffer:
    """Maintains per-term history buffers matching IsaacLab's layout.

    Mirrors IsaacLab's ``CircularBuffer`` semantics: at reset the buffer is
    empty, and the first ``append`` after reset broadcast-fills all
    ``HISTORY_LEN`` slots with the first observation (see IsaacLab
    ``circular_buffer.py``: ``buffer`` returns the full history with the
    first sample replicated until the buffer fills naturally).

    Without this priming we would inject ``HISTORY_LEN-1`` zeroed frames
    into the proprioception, which is OOD for any policy trained with
    history.
    """

    def __init__(self):
        self.gravity_hist = collections.deque(maxlen=HISTORY_LEN)
        self.angvel_hist = collections.deque(maxlen=HISTORY_LEN)
        self.jpos_hist = collections.deque(maxlen=HISTORY_LEN)
        self.jvel_hist = collections.deque(maxlen=HISTORY_LEN)
        self.action_hist = collections.deque(maxlen=HISTORY_LEN)
        self._primed = False

    def reset(self):
        self.gravity_hist.clear()
        self.angvel_hist.clear()
        self.jpos_hist.clear()
        self.jvel_hist.clear()
        self.action_hist.clear()
        self._primed = False

    def append(self, gravity, angvel, jpos_rel, jvel, action):
        g = gravity.astype(np.float32)
        a = angvel.astype(np.float32)
        jp = jpos_rel.astype(np.float32)
        jv = jvel.astype(np.float32)
        ac = action.astype(np.float32)
        if not self._primed:
            for _ in range(HISTORY_LEN):
                self.gravity_hist.append(g)
                self.angvel_hist.append(a)
                self.jpos_hist.append(jp)
                self.jvel_hist.append(jv)
                self.action_hist.append(ac)
            self._primed = True
        else:
            self.gravity_hist.append(g)
            self.angvel_hist.append(a)
            self.jpos_hist.append(jp)
            self.jvel_hist.append(jv)
            self.action_hist.append(ac)

    def get_flat(self) -> np.ndarray:
        """Return 990-dim proprioception in IsaacLab term-by-term layout.

        Term order MUST match ``PolicyCfg`` dataclass attribute order:
            base_ang_vel, joint_pos, joint_vel, actions, gravity_dir.
        Within each term, frames are oldest-first (CircularBuffer convention).
        """
        parts = []
        for hist in [self.angvel_hist, self.jpos_hist, self.jvel_hist,
                     self.action_hist, self.gravity_hist]:
            for frame in hist:
                parts.append(frame)
        return np.concatenate(parts).astype(np.float32)


def build_tokenizer_obs(motion_data, current_time, base_quat_wxyz, motion_fps):
    """Build 680-dim tokenizer input matching IsaacLab's exact layout.

    Training layout (from command_multi_future + motion_anchor_ori_b_mf):
      command_flat = cat([jpos_flat(10*31=310), jvel_flat(10*31=310)]) = 620
      command_nonflat = command_flat.reshape(10, 62)
      ori_nonflat = 6D_rot_diff per frame, shape (10, 6)
      encoder_input = cat(command_nonflat, ori_nonflat, dim=-1) = (10, 68)
      flattened to 680
    """
    m = _m(motion_data)
    total_frames = m["dof"].shape[0]
    dt = 1.0 / motion_fps

    cur_rot = Rot.from_quat([base_quat_wxyz[1], base_quat_wxyz[2],
                              base_quat_wxyz[3], base_quat_wxyz[0]])

    # Collect 10 future frames of jpos and jvel (in IsaacLab DOF order)
    jpos_frames = []  # each (31,) in IL order
    jvel_frames = []  # each (31,) in IL order
    ori_frames = []   # each (6,) 6D rotation diff

    # NOTE: future window starts at the CURRENT motion frame (t + 0.0), then
    # steps forward by DT_FUTURE_REF. IsaacLab uses
    # ``arange(num_future_frames) * frame_skips`` as the offset (see
    # ``TrackingCommand.future_time_steps_init`` in
    # ``gear_sonic/envs/manager_env/mdp/commands.py:354``), so frame 0 is the
    # robot's current motion frame, NOT the first future frame at t + 0.1.
    # An earlier version of this loop used ``(f + 1) * DT_FUTURE_REF`` and
    # was off by one.
    for f in range(NUM_FUTURE_FRAMES):
        future_time = current_time + f * DT_FUTURE_REF
        fi = min(int(future_time / dt), total_frames - 1)

        jpos_il = m["dof"][fi][IL_TO_MJ_DOF]
        jpos_frames.append(jpos_il.astype(np.float32))

        prev_fi = max(0, fi - 1)
        jvel_mj = (m["dof"][fi] - m["dof"][prev_fi]) * motion_fps
        jvel_il = jvel_mj[IL_TO_MJ_DOF]
        jvel_frames.append(jvel_il.astype(np.float32))

        # root_rot is xyzw in the PKL (scipy convention)
        fq = m["root_rot"][fi]
        future_rot = Rot.from_quat(fq)
        relative = cur_rot.inv() * future_rot
        rot_mat = relative.as_matrix()
        # 6D rotation representation MUST match IsaacLab's
        # ``commands.py::root_rot_dif_l_multi_future``:
        #   mat[..., :2].reshape(-1)
        # which flattens the (3, 2) slice in **row-major** order:
        #   [m00, m01, m10, m11, m20, m21]
        # An earlier version used ``concatenate([col0, col1])`` which produces
        # column-major [m00, m10, m20, m01, m11, m21] — same numbers, scrambled
        # order. That permutation made the policy see a fictitious large yaw
        # error at every step and "correct" by spinning the robot ~180° within
        # 1-2 seconds (visible in /tmp/h200_iter761_traj/{relaxed,walkforward}.csv
        # — relaxed_walk drifted +175° in 2s before this fix).
        ori_6d = rot_mat[:, :2].reshape(-1).astype(np.float32)
        ori_frames.append(ori_6d)

    # Replicate IsaacLab's layout: cat([all_jpos_flat, all_jvel_flat]).reshape(10, 62)
    jpos_flat = np.concatenate(jpos_frames)  # (310,)
    jvel_flat = np.concatenate(jvel_frames)  # (310,)
    command_flat = np.concatenate([jpos_flat, jvel_flat])  # (620,)
    command_nonflat = command_flat.reshape(NUM_FUTURE_FRAMES, -1)  # (10, 62)

    ori_nonflat = np.stack(ori_frames)  # (10, 6)

    encoder_input = np.concatenate([command_nonflat, ori_nonflat], axis=-1)  # (10, 68)
    return encoder_input.reshape(-1).astype(np.float32)  # (680,)


