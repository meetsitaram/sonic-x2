#!/usr/bin/env python3
"""Evaluate a trained SONIC checkpoint for the X2 Ultra in MuJoCo.

Loads a .pt checkpoint, reconstructs the universal-token actor (g1 encoder
+ g1_dyn decoder), reads the motion-lib PKL for reference commands, and
runs a PD-control loop in MuJoCo with on-screen rendering.

✅ PARITY VERIFIED — 2026-05-01 (iter-2000 sphere-feet checkpoint)
   The :class:`UniversalTokenActor` reimplementation below (SimpleMLP
   encoder + decoder + hand-rolled FSQ) matches the live
   ``UniversalTokenModule`` to float32 precision when fed identical
   inputs. Stage-by-stage check against a fresh
   ``dump_isaaclab_step0`` dump from the live IsaacLab module:

     - encoder latent: max|diff| = 2.4e-7
     - FSQ output:     max|diff| = 0  (exact)
     - final action:   max|diff| = 3.6e-7  rad

   So this script is a faithful PyTorch evaluator of the deployed
   policy — its bench numbers reflect the actual deployed actor, not
   a divergent re-implementation.

   Note (history): an earlier version of this docstring claimed a
   ~1.3 rad divergence. That conclusion was an artifact of a STALE
   diagnostic dump (from a different, older 16k-mesh checkpoint) being
   compared against weights from a newer checkpoint. Always regenerate
   the dump from the same .pt checkpoint you are evaluating — see
   ``gear_sonic/scripts/dump_isaaclab_step0.py``.

The robot is reset to the motion's frame-``--init-frame`` state using
Reference State Initialization (RSI) — the same teleport-into-the-motion
trick that IsaacLab does at every episode reset. RSI sets joint pos+vel
and root pos+quat+lin_vel+ang_vel from the motion, then physics takes
over and the policy must keep tracking.

Whenever the robot falls (pelvis drops below ``--fall-height``) or tips
over (gravity_body z > ``--fall-tilt-cos``) the episode is auto-reset to
the same RSI state — same as IsaacLab's termination/reset cycle. This
lets the policy keep getting fresh in-distribution starts during a long
viewer session.

Usage:
    python gear_sonic/scripts/eval_x2_mujoco.py \\
        --checkpoint logs_rl/.../model_step_006000.pt \\
        --motion gear_sonic/data/motions/x2_ultra_walk_forward.pkl

    Optional:
      --init-frame N         RSI motion frame (default 0)
      --fall-height 0.4      Pelvis z below this (m) triggers a reset
      --fall-tilt-cos -0.3   gravity_body[z] above this triggers a reset
                             (i.e. body tilted >~70° off upright)
      --max-episode 10.0     Force-reset after this many seconds (s)

Controls:
    SPACE - Pause / resume
    R     - Manually reset
    V     - Toggle camera tracking / free camera
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
import torch
import torch.nn as nn
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


# ---------- Actor ----------
class SimpleMLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.SiLU())
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)


def fsq_quantize(z, levels=32):
    """Replicate vector_quantize_pytorch.FSQ with uniform `levels`-per-dim.

    Matches FSQ.bound() exactly:
        half_l = (L-1) * (1+eps) / 2
        offset = 0.5 if L%2==0 else 0
        shift  = atanh(offset / half_l)
        bounded = tanh(z + shift) * half_l - offset
        return round(bounded) / (L // 2)

    The training config uses num_fsq_levels=32, fsq_level_list=32, so dim==
    effective_codebook_dim and FSQ.project_in/out are nn.Identity (no params).
    """
    L = float(levels)
    eps = 1e-3
    half_l = (L - 1.0) * (1.0 + eps) / 2.0
    offset = 0.5 if int(levels) % 2 == 0 else 0.0
    shift = math.atanh(offset / half_l) if offset != 0.0 else 0.0
    bounded = torch.tanh(z + shift) * half_l - offset
    return torch.round(bounded) / (int(levels) // 2)


class UniversalTokenActor(nn.Module):
    """Reproduces gear_sonic.trl.modules.universal_token_modules.UniversalTokenModule
    inference path for a single g1 encoder + g1_dyn decoder + FSQ quantizer.

    ✅ Verified to match the live UniversalTokenModule to ~3.6e-7 rad on the
    iter-2000 sphere-feet checkpoint (see module docstring). Validated by
    ``gear_sonic/scripts/dump_isaaclab_step0.py`` + a stage-by-stage
    encoder/FSQ/decoder bisection.

    Pipeline (matches universal_token_modules.encode + decode for g1/g1_dyn):
        tokenizer_obs(680) ── encoder MLP ──► 64
                          ── reshape(B, 2, 32) ──► FSQ(quantize) ──► (B, 2, 32)
                          ── flatten ──► token_flattened (B, 64)
        cat([token_flattened, proprioception(990)]) ──► decoder MLP ──► action(31)

    FSQ is a fixed deterministic bound+round (32 levels per dim) — no params.
    """

    MAX_NUM_TOKENS = 2
    TOKEN_DIM = 32
    FSQ_LEVELS = 32

    def __init__(self):
        super().__init__()
        self.encoder = SimpleMLP([680, 2048, 1024, 512, 512, 64])
        self.decoder = SimpleMLP([1054, 2048, 2048, 1024, 1024, 512, 512, 31])
        self.std = nn.Parameter(torch.zeros(31))

    def forward(self, proprioception, tokenizer_obs):
        # Encoder produces 64-dim continuous latent
        latent = self.encoder(tokenizer_obs)
        # Reshape to (B, num_tokens, token_dim) = (B, 2, 32)
        latent = latent.view(*latent.shape[:-1], self.MAX_NUM_TOKENS, self.TOKEN_DIM)
        # FSQ quantize (bound + round) — same shape
        quantized = fsq_quantize(latent, levels=self.FSQ_LEVELS)
        # Flatten back to token_flattened (B, 64)
        token_flat = quantized.view(*quantized.shape[:-2], -1)
        decoder_input = torch.cat([token_flat, proprioception], dim=-1)
        return self.decoder(decoder_input)


def load_actor_from_checkpoint(ckpt_path: str, device: str = "cpu"):
    # Full training checkpoints pickle class refs from the *public* ``trl``
    # package (HuggingFace TRL) present in the trainer's node env but not
    # necessarily here. We only need ``policy_state_dict`` (plain tensors), so
    # unpickle with a tolerant Unpickler that stubs out any class it can't
    # import -- the optimizer/args/env objects become inert stubs we never
    # touch, and no ``trl`` install is required to run the policy.
    import pickle as _pickle, types as _types

    class _Stub:
        def __init__(self, *a, **k): pass
        def __setstate__(self, state): pass

    class _StubUnpickler(_pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except Exception:
                return _Stub

    _tolerant = _types.ModuleType("_tolerant_pickle")
    for _a in dir(_pickle):
        setattr(_tolerant, _a, getattr(_pickle, _a))
    _tolerant.Unpickler = _StubUnpickler

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False,
                      pickle_module=_tolerant)
    sd = ckpt.get("policy_state_dict") or ckpt.get("actor_model_state_dict")
    if sd is None:
        raise KeyError("Cannot find policy state dict in checkpoint")
    actor = UniversalTokenActor()
    new_sd = {}
    for k, v in sd.items():
        if k == "std":
            new_sd["std"] = v
        elif k.startswith("actor_module.encoders.g1.module."):
            new_sd[k.replace("actor_module.encoders.g1.module.", "encoder.module.")] = v
        elif k.startswith("actor_module.decoders.g1_dyn.module."):
            new_sd[k.replace("actor_module.decoders.g1_dyn.module.", "decoder.module.")] = v
    actor.load_state_dict(new_sd)
    actor.eval()
    return actor.to(device)


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


def load_playlist_motion_data(playlist_path):
    """Build a single-clip motion-lib dict from a warehouse playlist YAML.

    Reuses ``_warehouse_playlist.build_concat`` so the in-memory dict is
    bit-identical to ``make_warehouse_motion.py --out`` for the same YAML
    (verified by the stitcher's ``--check-runtime-parity`` flag). Lets us
    iterate on the playlist YAML without re-running the stitcher CLI.
    """
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from _warehouse_playlist import build_concat, load_playlist  # type: ignore
    pl = load_playlist(_Path(playlist_path))
    motion = build_concat(pl)
    return {pl.output_key: motion}


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


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    motion_grp = parser.add_mutually_exclusive_group(required=True)
    motion_grp.add_argument("--motion", help="Single-clip motion-lib PKL.")
    motion_grp.add_argument(
        "--playlist",
        help="Warehouse playlist YAML (resolved via _warehouse_playlist."
             "build_concat into a virtual single-clip motion). Mutually "
             "exclusive with --motion.",
    )
    motion_grp.add_argument(
        "--motions",
        help="Multi-clip motion-lib PKL: iterate EVERY clip one-by-one in one "
             "viewer session, RSI-resetting the robot before each clip. Advances "
             "to the next clip when the current one completes a full pass OR the "
             "robot falls. Mutually exclusive with --motion / --playlist.",
    )
    parser.add_argument(
        "--loop-clips", action="store_true",
        help="With --motions, loop back to the first clip after the last "
             "(default: stop after one pass through all clips).")
    parser.add_argument(
        "--seconds-per-clip", type=float, default=0.0,
        help="With --motions, show each clip for at most this many seconds before "
             "auto-advancing (default 0 = play the full clip once). E.g. 5 = a 5 s "
             "snippet of each. Keyboard N/B still override to step manually.")
    parser.add_argument(
        "--debug-obs", action="store_true",
        help="Print the per-step obs/action debug block every 250 steps "
             "(old default). Off by default so clip titles, [end] lines and "
             "★ RATING lines stay readable.")
    parser.add_argument(
        "--clip-pause", type=float, default=0.0,
        help="With --motions, hold the sim frozen this many seconds between "
             "clips (robot freezes at its final pose, terminal shows which clip "
             "just ended). Rating keys E/M/D still apply to the finished clip "
             "during the pause; N skips the wait. E.g. 15 for demo triage.")
    parser.add_argument(
        "--clip", default=None,
        help="With --motions, only play clips whose name CONTAINS this substring "
             "(case-insensitive). E.g. --clip dance_disco_fever_001__A465 plays just "
             "that one dance; --clip freedom_wheels plays all freedom_wheels variants.")
    parser.add_argument(
        "--record", default=None,
        help="Capture the per-step sim trajectory (MuJoCo-order measured joint "
             "pos/vel + commanded target) to this .npz and EXIT after one clip pass. "
             "For sim-vs-real comparison against the LeRobot capture.")
    parser.add_argument(
        "--record-seconds", type=float, default=0.0,
        help="With --record, capture this many seconds (default 0 = one full clip pass).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--speed", type=float, default=1.0)
    # Wrist handling in deploy. The policy's wrist actions diverge in the
    # MuJoCo deploy loop (the wrist is the weakest joint; explicit-PD tracking
    # error corrupts the proprioception obs -> action runaway -> wrist slams to
    # its limit), even though the SAME policy keeps wrists natural in IsaacLab.
    # Taking the wrists out of the policy loop removes the divergence seed.
    # DEFAULT = freeze wrists (interim mitigation, 2026-07-13): the policy's
    # wrist actions diverge in deploy, so by default we hold the wrists out of
    # the loop. --wrist-ref articulates them from the reference instead;
    # --no-wrist-fix restores the old raw-policy-wrist behavior.
    wgrp = parser.add_mutually_exclusive_group()
    wgrp.add_argument("--freeze-wrist", action="store_true",
                      help="Hold wrists at the neutral (default) pose (this is the default).")
    wgrp.add_argument("--wrist-ref", action="store_true",
                      help="Drive wrists to the reference motion's wrist angles each "
                           "frame (keeps articulation, no divergence).")
    wgrp.add_argument("--no-wrist-fix", action="store_true",
                      help="Restore raw policy control of the wrists (they will diverge/twist).")
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
        help="If pelvis world z drops below this (m), auto-reset (default 0.4).",
    )
    parser.add_argument(
        "--fall-tilt-cos",
        type=float,
        default=-0.3,
        help="If gravity in body frame's z component goes above this, auto-reset. "
        "-1.0 = perfectly upright, 0.0 = horizontal. Default -0.3 ~ 72° tilt.",
    )
    parser.add_argument(
        "--max-episode",
        type=float,
        default=0.0,
        help="If > 0, force a reset after this many simulated seconds (default 0 = no limit).",
    )
    parser.add_argument(
        "--push-force", type=float, default=200.0,
        help="Magnitude (N) of the manual forward push applied on key 'P' "
             "(adjust live with '[' / ']'). Default 200.")
    parser.add_argument(
        "--push-duration", type=float, default=0.2,
        help="How long (s) the manual push force is held. Default 0.2.")
    parser.add_argument(
        "--push-body", default="head_pitch_link",
        help="Body the manual push is applied to. Default head_pitch_link "
             "(head/neck) -- it sits ~0.40 m above the waist pitch joint, giving "
             "the largest moment arm about the waist, so it actually stresses "
             "waist-pitch control. A chest/pelvis push is at/below the waist and "
             "mostly loads ankles/hips instead. Use --push-body head_yaw_link "
             "(neck) or pelvis to compare.")
    parser.add_argument(
        "--assist-force", type=float, default=0.0,
        help="Sustained external ASSIST force (N) applied every step at --push-body "
             "(set --push-body torso_link for a chest assist), directed UP + toward "
             "the robot's BACK -- mimics a human helping the robot stand from lying. "
             "Default 0 (off).")
    parser.add_argument("--assist-up", type=float, default=1.0,
                        help="Vertical (up) weight of the assist direction. Default 1.0.")
    parser.add_argument("--assist-back", type=float, default=1.0,
                        help="Backward (toward robot's back) weight of the assist direction. Default 1.0.")

    # ---- Per-joint-group PD scaling. Multiplicative on top of the
    # DEPLOYMENT_KP_SCALE / DEPLOYMENT_KD_SCALE tables baked into this script
    # from G16b. Use these flags to A/B against the baked defaults at runtime
    # (e.g. --kp-scale-ankle 0.6667 undoes the baked ankle ×1.5 to recover the
    # pre-G16b training-equivalent PD).
    parser.add_argument("--kp-scale", type=float, default=1.0,
                        help="Global multiplier applied to KP on every DOF.")
    parser.add_argument("--kd-scale", type=float, default=1.0,
                        help="Global multiplier applied to KD on every DOF.")
    for group, default_help in [
        ("leg",   "hip_yaw/roll/pitch"),
        ("knee",  "knee"),
        ("ankle", "ankle_pitch/roll"),
        ("waist", "waist_yaw/pitch/roll"),
        ("arm",   "shoulder + elbow"),
        ("wrist", "wrist_yaw/pitch/roll"),
        ("head",  "head_yaw/pitch"),
    ]:
        parser.add_argument(f"--kp-scale-{group}", type=float, default=1.0,
                            help=f"KP multiplier on {group} group "
                                 f"({default_help}). Default 1.0.")
        parser.add_argument(f"--kd-scale-{group}", type=float, default=1.0,
                            help=f"KD multiplier on {group} group "
                                 f"({default_help}). Default 1.0.")
    args = parser.parse_args()

    # Build per-DOF KP/KD scale arrays from the CLI groups, apply to the
    # module-level KP/KD imported by the rest of this script.
    _PD_GROUPS = [
        ("hip", "leg"), ("knee", "knee"), ("ankle", "ankle"),
        ("waist", "waist"), ("shoulder", "arm"), ("elbow", "arm"),
        ("wrist", "wrist"), ("head", "head"),
    ]
    group_kp = {"leg": args.kp_scale_leg, "knee": args.kp_scale_knee,
                "ankle": args.kp_scale_ankle, "waist": args.kp_scale_waist,
                "arm": args.kp_scale_arm, "wrist": args.kp_scale_wrist,
                "head": args.kp_scale_head}
    group_kd = {"leg": args.kd_scale_leg, "knee": args.kd_scale_knee,
                "ankle": args.kd_scale_ankle, "waist": args.kd_scale_waist,
                "arm": args.kd_scale_arm, "wrist": args.kd_scale_wrist,
                "head": args.kd_scale_head}
    kp_runtime = np.full(NUM_DOFS, float(args.kp_scale), dtype=np.float64)
    kd_runtime = np.full(NUM_DOFS, float(args.kd_scale), dtype=np.float64)
    for i, jname in enumerate(MUJOCO_JOINT_NAMES):
        short = jname.replace("left_", "").replace("right_", "").replace("_joint", "")
        for tok, grp in _PD_GROUPS:
            if tok in short:
                kp_runtime[i] *= group_kp[grp]
                kd_runtime[i] *= group_kd[grp]
                break
    summary = []
    if abs(args.kp_scale - 1.0) > 1e-9: summary.append(f"global kp×{args.kp_scale:g}")
    if abs(args.kd_scale - 1.0) > 1e-9: summary.append(f"global kd×{args.kd_scale:g}")
    for grp in group_kp:
        if abs(group_kp[grp]-1.0)>1e-9 or abs(group_kd[grp]-1.0)>1e-9:
            summary.append(f"{grp}(kp×{group_kp[grp]:g},kd×{group_kd[grp]:g})")
    if summary:
        print(f"  Runtime PD overrides (multiplicative on top of baked "
              f"DEPLOYMENT_*_SCALE): {', '.join(summary)}", flush=True)
    global KP, KD
    KP = (KP * kp_runtime).astype(np.float64)
    KD = (KD * kd_runtime).astype(np.float64)

    print(f"Loading actor from {args.checkpoint} ...", flush=True)
    actor = load_actor_from_checkpoint(args.checkpoint, args.device)
    print("  Actor loaded.", flush=True)

    # ---- Build the clip list: single (--motion/--playlist) or multi (--motions) ----
    if args.motions is not None:
        print(f"Loading multi-clip motions from {args.motions} ...", flush=True)
        _all = load_motion_data(args.motions)
        clip_names = list(_all.keys())
        if args.clip:
            clip_names = [k for k in clip_names if args.clip.lower() in k.lower()]
            if not clip_names:
                raise SystemExit(
                    f"[eval] --clip '{args.clip}' matched no clip names in {args.motions}")
        clips = [{k: _all[k]} for k in clip_names]
        _sel = f" matching '{args.clip}'" if args.clip else ""
        print(f"  {len(clips)} clip(s){_sel} -- iterating one-by-one, RSI-reset before each.",
              flush=True)
    elif args.playlist is not None:
        print(f"Loading playlist from {args.playlist} ...", flush=True)
        _md = load_playlist_motion_data(args.playlist)
        clip_names, clips = [list(_md.keys())[0]], [_md]
    else:
        print(f"Loading motion from {args.motion} ...", flush=True)
        _md = load_motion_data(args.motion)
        clip_names, clips = [list(_md.keys())[0]], [_md]
    n_clips = len(clips)

    print("Loading MuJoCo model ...", flush=True)
    mj_model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = SIM_DT

    pelvis_id = mj_model.body("pelvis").id
    push_body_id = mj_model.body(args.push_body).id
    push_steps_left = 0
    push_force = float(args.push_force)
    pending_clip_jump = 0      # +1 next / -1 previous, set from key_callback
    pending_rating = None      # 'E'/'M'/'D' = rate current dance easy/medium/difficult

    # ---- Per-clip state, reassigned by _load_clip() when iterating clips.
    # RSI: pull the full robot reset state from motion frame `init_frame`.
    init_frame = int(args.init_frame)
    clip_idx = 0
    motion_data = None
    total_frames = 0
    motion_fps = 30.0
    init_motion_state = None
    init_root_z = 0.0

    def _load_clip(idx):
        nonlocal motion_data, total_frames, motion_fps, init_motion_state, init_root_z
        motion_data = clips[idx]
        total_frames = get_total_frames(motion_data)
        motion_fps = get_motion_fps(motion_data)
        init_motion_state = compute_motion_state(motion_data, init_frame, motion_fps)
        init_root_z = float(init_motion_state["root_pos_w"][2])
        if n_clips > 1:
            print(f"\n=== DANCE {idx + 1}/{n_clips}: {clip_names[idx]} "
                  f"({total_frames / motion_fps:.1f}s) ===", flush=True)
        else:
            print(f"\n[RSI] {clip_names[idx]}: {total_frames} frames @ {motion_fps:g} fps "
                  f"= {total_frames / motion_fps:.1f}s (RSI frame {init_frame})", flush=True)

    _load_clip(0)

    print(f"  Default DOF (MJ order): {DEFAULT_DOF}", flush=True)
    print(f"  Action scale (MJ order): {ACTION_SCALE}", flush=True)

    prop_buf = ProprioceptionBuffer()
    last_action_mj = np.zeros(NUM_DOFS, dtype=np.float32)
    _rec = {"t": [], "frame": [], "qpos": [], "qvel": [], "target": []} if args.record else None
    sim_time = float(init_frame) / motion_fps
    step_count = 0
    episode_count = 0
    episode_start_step = 0
    paused = False

    def _apply_init_state():
        """Teleport the robot to the RSI motion frame state.

        Mirrors IsaacLab's ``write_joint_state_to_sim`` /
        ``write_root_state_to_sim`` at episode reset.
        """
        s = init_motion_state
        mj_data.qpos[0] = 0.0  # discard motion world XY offset
        mj_data.qpos[1] = 0.0
        mj_data.qpos[2] = float(s["root_pos_w"][2])
        mj_data.qpos[3:7] = s["root_quat_w_wxyz"]
        mj_data.qpos[7:7 + NUM_DOFS] = s["joint_pos_mj"]
        # MuJoCo free-joint qvel convention (verified empirically; see commit msg):
        #   qvel[0:3]  -> WORLD-frame linear velocity
        #   qvel[3:6]  -> BODY-LOCAL-frame angular velocity (NOT world!)
        # Motion lib stores both in world frame, so the angular component must
        # be rotated into the body frame before writing.
        mj_data.qvel[0:3] = s["root_lin_vel_w"]
        mj_data.qvel[3:6] = quat_rotate_inverse(
            s["root_quat_w_wxyz"], s["root_ang_vel_w"]
        )
        mj_data.qvel[6:6 + NUM_DOFS] = s["joint_vel_mj"]
        mj_data.xfrc_applied[:] = 0
        mujoco.mj_forward(mj_model, mj_data)

    _apply_init_state()

    def reset_state(reason=""):
        nonlocal sim_time, last_action_mj, episode_count, episode_start_step
        sim_time = float(init_frame) / motion_fps
        last_action_mj[:] = 0
        prop_buf.reset()
        _apply_init_state()
        episode_count += 1
        episode_start_step = step_count
        tag = f" ({reason})" if reason else ""
        print(f"\n[reset]{tag} starting episode {episode_count}", flush=True)

    def key_callback(keycode):
        nonlocal paused, push_steps_left, push_force, pending_clip_jump, pending_rating
        import glfw
        if keycode == glfw.KEY_SPACE:
            paused = not paused
            print("Paused" if paused else "Resumed", flush=True)
        elif keycode == glfw.KEY_R:
            reset_state("manual")
        elif keycode == glfw.KEY_N:          # next dance
            pending_clip_jump = 1
        elif keycode == glfw.KEY_B:          # back / previous dance
            pending_clip_jump = -1
        elif keycode == glfw.KEY_E:          # rate current dance: easy
            pending_rating = "easy"
        elif keycode == glfw.KEY_M:          # rate current dance: medium
            pending_rating = "medium"
        elif keycode == glfw.KEY_D:          # rate current dance: difficult
            pending_rating = "difficult"
        elif keycode == glfw.KEY_P:
            push_steps_left = max(1, int(round(args.push_duration / CONTROL_DT)))
            print(f"[PUSH] forward {push_force:.0f} N for {args.push_duration:.2f}s "
                  f"on {args.push_body}", flush=True)
        elif keycode == glfw.KEY_RIGHT_BRACKET:      # ']' -> stronger
            push_force += 50.0
            print(f"[PUSH] magnitude -> {push_force:.0f} N", flush=True)
        elif keycode == glfw.KEY_LEFT_BRACKET:       # '[' -> weaker
            push_force = max(0.0, push_force - 50.0)
            print(f"[PUSH] magnitude -> {push_force:.0f} N", flush=True)
        elif keycode == glfw.KEY_V:
            if viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = pelvis_id

    print("\n=== X2 MuJoCo Eval ===", flush=True)
    print(f"Robot RSI-initialized from motion frame {init_frame}.", flush=True)
    print(f"Auto-reset triggers: pelvis_z < {args.fall_height:.2f} m, "
          f"or gravity_body[z] > {args.fall_tilt_cos:.2f} (~tilt > "
          f"{int(np.rad2deg(np.arccos(-args.fall_tilt_cos)))}°).", flush=True)
    if args.max_episode > 0:
        print(f"Max episode length: {args.max_episode:.1f} s.", flush=True)
    print("Keys: SPACE pause | R reset | N next | B back | E/M/D rate easy/med/hard | "
          "V camera | P push | [ / ] push force.\n", flush=True)

    with mujoco.viewer.launch_passive(
        mj_model, mj_data,
        key_callback=key_callback,
        show_left_ui=False, show_right_ui=False,
    ) as viewer:
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0.0, 0.0, init_root_z]
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = pelvis_id

        wall_start = time.time()
        # Motion-tracking sim_time is anchored to wall_start. Each reset
        # rebases wall_start so real-time pacing stays smooth across episodes.
        wall_start -= sim_time

        while viewer.is_running():
            if paused:
                viewer.sync()
                time.sleep(0.02)
                continue

            # -- Pending manual dance nav / mark (set from key_callback; handled
            #    here on the main loop for thread-safety) --
            if pending_rating is not None:
                print(f"  ★ RATING {pending_rating} | DANCE {clip_idx + 1}/{n_clips}: "
                      f"{clip_names[clip_idx]}", flush=True)
                pending_rating = None
            if pending_clip_jump != 0 and n_clips > 1:
                clip_idx = (clip_idx + pending_clip_jump) % n_clips
                pending_clip_jump = 0
                _load_clip(clip_idx)
                reset_state("manual nav")
                wall_start = time.time() - sim_time

            # -- Motion phase --
            motion_time = sim_time * args.speed
            motion_frame = int(motion_time * motion_fps) % total_frames
            motion_time = motion_frame / motion_fps

            # -- Read MuJoCo state --
            qpos_j = mj_data.qpos[7:7 + NUM_DOFS].copy()
            qvel_j = mj_data.qvel[6:6 + NUM_DOFS].copy()
            base_quat = mj_data.qpos[3:7].copy()  # wxyz
            # MuJoCo free-joint qvel[3:6] is in BODY-LOCAL frame (verified
            # empirically via quaternion integration). IsaacLab's
            # ``base_ang_vel`` proprioception term is also body-local, so we
            # consume qvel[3:6] directly — no rotation. Previously we
            # quat_rotate_inverse'd it under the wrong assumption that it was
            # world-frame, which caused a double rotation and corrupted the
            # angular-velocity history every step.
            base_angvel = mj_data.qvel[3:6].copy()

            # Convert MJ→IL using IL_TO_MJ_DOF as gather index:
            #   result[il_pos] = source[IL_TO_MJ_DOF[il_pos]]
            dof_pos_il = qpos_j[IL_TO_MJ_DOF]
            dof_vel_il = qvel_j[IL_TO_MJ_DOF]
            action_il = last_action_mj[IL_TO_MJ_DOF]

            gravity = quat_rotate_inverse(base_quat, np.array([0., 0., -1.]))
            dof_pos_rel_il = dof_pos_il - DEFAULT_DOF[IL_TO_MJ_DOF]

            # -- Build observation + run policy --
            prop_buf.append(gravity, base_angvel, dof_pos_rel_il, dof_vel_il, action_il)
            proprioception = prop_buf.get_flat()
            tokenizer_obs = build_tokenizer_obs(
                motion_data, motion_time, base_quat, motion_fps)

            with torch.no_grad():
                prop_t = torch.from_numpy(proprioception).unsqueeze(0).to(args.device)
                tok_t = torch.from_numpy(tokenizer_obs).unsqueeze(0).to(args.device)
                action_il_t = actor(prop_t, tok_t).squeeze(0).cpu().numpy()

            # Match the IsaacLab wrapper, which clips env_actions to
            # ±action_clip_value BEFORE applying them AND before the action
            # manager stores them (so the `last_action` history obs is also
            # clipped). Without this the deploy applies + re-observes the raw
            # action, which can run away (wrist actions -> 60+ -> target far
            # past the joint limit -> wrist slams to its stop). ACTION_CLIP=0
            # disables (old behavior) for A/B testing.
            if ACTION_CLIP > 0:
                action_il_t = np.clip(action_il_t, -ACTION_CLIP, ACTION_CLIP)

            # Take the wrists out of the (deploy-divergent) policy loop: zero the
            # wrist action channels so the last_action obs for the wrist is 0 and
            # the wrist target comes from below, not the runaway-prone policy
            # output. Verified 2026-07-13 to stop the wrist-twist. (Env
            # FREEZE_WRIST kept as an override for A/B diagnostics.)
            _do_wrist_fix = (not args.no_wrist_fix) and not os.environ.get("NO_WRIST_FIX")
            if _do_wrist_fix:
                action_il_t[_WRIST_IL_IDX] = 0.0

            # Convert IL→MJ using MJ_TO_IL_DOF as gather index:
            #   result[mj_pos] = source[MJ_TO_IL_DOF[mj_pos]]
            action_mj = action_il_t[MJ_TO_IL_DOF]
            last_action_mj = action_mj.copy()
            target_pos = DEFAULT_DOF + action_mj * ACTION_SCALE

            # --wrist-ref: articulate the wrists from the reference motion
            # instead of holding them at neutral (--freeze-wrist).
            if args.wrist_ref:
                ref_dof_mj = np.asarray(_m(motion_data)["dof"][motion_frame], dtype=np.float64)
                target_pos[_WRIST_MJ_IDX] = ref_dof_mj[_WRIST_MJ_IDX]

            # -- Trajectory capture (--record): measured joint state (pre-step) +
            #    commanded target, MuJoCo order, one sample per 50 Hz control tick. --
            if _rec is not None:
                _rec["t"].append(sim_time)
                _rec["frame"].append(motion_frame)
                _rec["qpos"].append(qpos_j.copy())
                _rec["qvel"].append(qvel_j.copy())
                _rec["target"].append(target_pos.copy())
                _rec.setdefault("root", []).append(mj_data.qpos[0:7].copy())

            # -- Manual forward push (key 'P'): external force at the head/neck
            # (--push-body, default head_pitch_link -- high up for a large moment
            # arm about the waist), applied along the robot's own forward (+X
            # body) projected to horizontal, so "forward" is relative to facing. --
            if push_steps_left > 0:
                w, x, y, z = base_quat
                u = np.array([x, y, z])
                fwd = np.array([1.0, 0.0, 0.0])
                fwd_w = fwd + 2.0 * w * np.cross(u, fwd) + 2.0 * np.cross(u, np.cross(u, fwd))
                fwd_w[2] = 0.0
                nrm = np.linalg.norm(fwd_w)
                if nrm > 1e-6:
                    fwd_w /= nrm
                mj_data.xfrc_applied[push_body_id, :3] = push_force * fwd_w
            else:
                mj_data.xfrc_applied[push_body_id, :3] = 0.0

            # -- Sustained ASSIST force (--assist-force): lift UP + toward the
            # robot's BACK, at --push-body (torso_link = chest). Mimics a human
            # helping the robot up from lying. Added on top of any P-push. --
            if args.assist_force > 0.0:
                w2, x2, y2, z2 = base_quat
                u2 = np.array([x2, y2, z2])
                fwd2 = np.array([1.0, 0.0, 0.0])
                fwd2_w = fwd2 + 2.0 * w2 * np.cross(u2, fwd2) + 2.0 * np.cross(u2, np.cross(u2, fwd2))
                fwd2_w[2] = 0.0
                n2 = np.linalg.norm(fwd2_w)
                if n2 > 1e-6:
                    fwd2_w /= n2
                assist_dir = args.assist_up * np.array([0.0, 0.0, 1.0]) + args.assist_back * (-fwd2_w)
                na = np.linalg.norm(assist_dir)
                if na > 1e-6:
                    assist_dir /= na
                mj_data.xfrc_applied[push_body_id, :3] += args.assist_force * assist_dir

            # -- PD control + step --
            for _ in range(DECIMATION):
                torque = KP * (target_pos - mj_data.qpos[7:7 + NUM_DOFS]) \
                       - KD * mj_data.qvel[6:6 + NUM_DOFS]
                for j in range(NUM_DOFS):
                    mj_data.ctrl[JOINT_TO_ACTUATOR[j]] = torque[j]
                mujoco.mj_step(mj_model, mj_data)

            _wrist_dump(step_count, target_pos, mj_data, action_mj)

            if push_steps_left > 0:
                push_steps_left -= 1
                if push_steps_left == 0:
                    mj_data.xfrc_applied[push_body_id, :3] = 0.0

            sim_time += CONTROL_DT
            step_count += 1
            viewer.sync()

            # -- --record: stop + dump after one clip pass (or --record-seconds). --
            if _rec is not None:
                _dur = (args.record_seconds if args.record_seconds > 0
                        else total_frames / motion_fps / max(args.speed, 1e-9))
                if len(_rec["t"]) * CONTROL_DT >= _dur:
                    jn = [mj_model.joint(i + 1).name for i in range(NUM_DOFS)]
                    np.savez(args.record,
                             joint_names=np.array(jn),
                             t=np.array(_rec["t"]),
                             motion_frame=np.array(_rec["frame"]),
                             qpos=np.array(_rec["qpos"]),
                             qvel=np.array(_rec["qvel"]),
                             target=np.array(_rec["target"]),
                             root=np.array(_rec.get("root", [])),
                             clip=(clip_names[clip_idx] if clip_names else ""),
                             fps=float(motion_fps))
                    print(f"[record] wrote {len(_rec['t'])} steps ({_dur:.1f}s) to "
                          f"{args.record}", flush=True)
                    break

            wall_elapsed = time.time() - wall_start
            if sim_time > wall_elapsed:
                time.sleep(sim_time - wall_elapsed)

            # -- Auto-reset on fall (IsaacLab-style termination) --
            pelvis_z = float(mj_data.qpos[2])
            grav_z = float(gravity[2])
            episode_seconds = (step_count - episode_start_step) * CONTROL_DT
            reset_reason = None
            if pelvis_z < args.fall_height:
                reset_reason = f"pelvis_z={pelvis_z:.3f} < {args.fall_height:.2f}"
            elif grav_z > args.fall_tilt_cos:
                reset_reason = (f"gravity_body[z]={grav_z:+.2f} > {args.fall_tilt_cos:.2f} "
                                f"(tilt {int(np.rad2deg(np.arccos(np.clip(-grav_z,-1,1))))}°)")
            elif args.max_episode > 0 and episode_seconds >= args.max_episode:
                reset_reason = f"reached --max-episode={args.max_episode:.1f}s"
            elif n_clips > 1:
                clip_window = total_frames / motion_fps / max(args.speed, 1e-9)
                if args.seconds_per_clip > 0:
                    clip_window = min(clip_window, args.seconds_per_clip)
                if episode_seconds >= clip_window:
                    reset_reason = "window done"
            if reset_reason is not None:
                print(f"  [end] DANCE {clip_idx + 1}/{n_clips} {clip_names[clip_idx]} "
                      f"ran {episode_seconds:.2f}s ({reset_reason})", flush=True)
                if args.clip_pause > 0 and n_clips > 1:
                    # Freeze at the final pose so the just-played clip can be
                    # rated/noted before the next one starts. E/M/D rate the
                    # finished clip; N cuts the pause short.
                    print(f"  [pause] holding {args.clip_pause:.0f}s after "
                          f"{clip_names[clip_idx]} (E/M/D to rate, N to skip wait)",
                          flush=True)
                    pause_end = time.time() + args.clip_pause
                    while viewer.is_running() and time.time() < pause_end:
                        if pending_rating is not None:
                            print(f"  ★ RATING {pending_rating} | DANCE "
                                  f"{clip_idx + 1}/{n_clips}: {clip_names[clip_idx]}",
                                  flush=True)
                            pending_rating = None
                        if pending_clip_jump != 0:
                            pending_clip_jump = 0
                            break
                        viewer.sync()
                        time.sleep(0.05)
                    if not viewer.is_running():
                        break
                if n_clips > 1:
                    # Advance to the next dance clip (RSI-reset happens in reset_state).
                    clip_idx += 1
                    if clip_idx >= n_clips:
                        if not args.loop_clips:
                            print(f"\n[done] iterated all {n_clips} clips.", flush=True)
                            break
                        clip_idx = 0
                    _load_clip(clip_idx)
                reset_state(reset_reason)
                wall_start = time.time() - sim_time
                continue

            if args.debug_obs and (step_count <= 5 or step_count % 250 == 0):
                h = mj_data.qpos[2]
                print(f"\n[ep {episode_count}] step={step_count}  sim={sim_time:.2f}s  "
                      f"frame={motion_frame}/{total_frames}  height={h:.3f}m",
                      flush=True)
                print(f"  gravity_body:  {gravity}", flush=True)
                print(f"  angvel_body:   {base_angvel}", flush=True)
                print(f"  base_quat:     {base_quat}", flush=True)
                print(f"  dof_pos_il[:6]: {dof_pos_il[:6]}", flush=True)
                print(f"  dof_vel_il[:6]: {dof_vel_il[:6]}", flush=True)
                print(f"  action_il[:6]:  {action_il_t[:6]}", flush=True)
                print(f"  action_mj[:6]:  {action_mj[:6]}", flush=True)
                print(f"  target_pos[:6]: {target_pos[:6]}", flush=True)
                print(f"  action |min|={np.abs(action_il_t).min():.4f} "
                      f"|max|={np.abs(action_il_t).max():.4f} "
                      f"|mean|={np.abs(action_il_t).mean():.4f}", flush=True)
                _top = np.argsort(np.abs(action_mj))[::-1][:4]
                print("  top|action| joints: " + ", ".join(
                    f"{MUJOCO_JOINT_NAMES[i].replace('_joint','')}={action_mj[i]:.1f}"
                    for i in _top), flush=True)
                print(f"  prop shape={proprioception.shape} "
                      f"|min|={proprioception.min():.4f} |max|={proprioception.max():.4f}",
                      flush=True)
                print(f"  tok  shape={tokenizer_obs.shape} "
                      f"|min|={tokenizer_obs.min():.4f} |max|={tokenizer_obs.max():.4f}",
                      flush=True)

    print("Viewer closed.")


if __name__ == "__main__":
    main()
