# sonic-x2

Standalone bundle for running the **SONIC whole-body tracking policy on the
AgiBot X2 Ultra humanoid (31 DOF) in MuJoCo** — the quick-play companion to
the deployment-ready X2 stack. Everything needed is in this repo: policy
(ONNX), robot model (MJCF + meshes), reference motions, the validated
MuJoCo player, and the real-deploy tuning presets. CPU-only.

## Kinematic reference vs SONIC (side by side)

Left: the reference motion played kinematically (poses written straight to
the sim, no physics). Right: the SONIC policy tracking that same reference
under full physics with the real robot's deploy tuning — both clips run to
`motion_end` with no falls.

**Relaxed walk** (`Relaxed_walk_forward_001__A057`, 14.8 s):

![Relaxed walk preview](media/relaxed_walk_preview.gif)

**Gangnam victory dance** (`victory_dance_gangam_style_rodeo_R_001__A324`,
9.2 s, from the gamepad demo bank, played under the `bigrun` demo preset):

![Gangnam dance preview](media/gangam_preview.gif)

*(10 s previews — full videos:
[walk side-by-side](media/relaxed_walk_side_by_side.mp4) ·
[dance side-by-side](media/gangam_side_by_side.mp4))*

> The `.mp4`s are tracked with **git LFS** — install `git-lfs` before
> cloning to get them. The GIF previews and everything else (model,
> motions, scripts) are plain git and work regardless.

Reproduce (needs ffmpeg on PATH; hstack the pairs with ffmpeg):

```bash
./play_relaxed_walk.sh --kinematic --record media/relaxed_walk_kinematic.mp4
./play_relaxed_walk.sh --no-viewer --max-episode 15 --record media/relaxed_walk_sonic.mp4
./play_gangam_dance.sh --kinematic --clip victory_dance_gangam_style_rodeo_R_001__A324 \
    --record media/gangam_kinematic.mp4
./play_gangam_dance.sh --no-viewer --max-episode 10 --record media/gangam_sonic.mp4
```

## Quickstart

```bash
./install.sh              # creates .venv (mujoco, onnxruntime, torch-cpu, scipy, joblib)
./play_relaxed_walk.sh    # MuJoCo viewer: SONIC walks the relaxed-walk clip
./play_idle_stand.sh      # idle stand (hands-on-back loop)
./play_gangam_dance.sh    # Gangnam-style victory dance from the gamepad demo bank
```

Useful variants (flags pass through to the player):

```bash
./play_relaxed_walk.sh --no-viewer --max-episode 20   # headless + tracking metrics
./play_gangam_dance.sh --speed 0.5                    # half-speed reference
./play_relaxed_walk.sh --tuning ''                    # raw training-parity gains
```

## Deploy-parity tuning (on by default)

The player applies the **real robot's deploy tuning** out of the box
(`configs/real_deploy_tuning/walk_101.yaml` — per-group PD trim, per-joint
target-deviation clamps, target low-pass filter, action clip), so what you
see in MuJoCo is the deployed behavior, not raw training gains. The dance
script uses `bigrun.yaml`, the demo preset the dances are deployed with.
Pass `--tuning ''` for training-parity gains, or point `--tuning` at any
preset in `configs/real_deploy_tuning/`.

## Contents

| Path | What |
|---|---|
| `models/x2_sonic_14000_g1.onnx` | SONIC X2 policy — the v7b fine-tune deploy candidate (step 14000), fused g1-encoder → FSQ → decoder, 1670-D obs → 31-D action. |
| `motions/x2_relaxed_walk.pkl` | `Relaxed_walk_forward_001__A057` (14.8 s @ 50 fps) — the demo relaxed walk. |
| `motions/x2_idle_stand.pkl` | Hands-on-back idle loop — the demo's idle-stand reference. |
| `motions/x2_gangam_dance.pkl` | `victory_dance_gangam_style_rodeo_R_001__A324` (+ mirror, 9.2 s @ 50 fps) — from the gamepad dance bank; in-distribution for this checkpoint. |
| `assets/mjcf/x2_ultra.xml` (+ `assets/urdf/x2_ultra/meshes/`) | X2 Ultra MuJoCo model. |
| `scripts/eval_x2_mujoco_onnx.py` | The player: ONNX policy at 50 Hz, deploy-tuned PD, RSI + fall detection, tracking metrics. |
| `scripts/eval_x2_mujoco.py` | Constants + observation construction the player builds on (shared with the deployment stack). |
| `configs/real_deploy_tuning/` | Robot tuning presets (`walk_101` default, `bigrun` demo config, and variants). |

## Notes

- Motion PKLs are joblib dicts `{clip: {root_trans_offset, root_rot, dof,
  pose_aa, fps}}` — `dof` is 31 joints in MJCF order; walk/dance clips are
  50 fps, the idle loop 30 fps (the player reads each clip's `fps`).
- Every episode prints survival time and joint/pelvis tracking error;
  a clean run ends with `motion_end`, not a fall.
- For the full deployment stack (gamepad runtime, teleop, planner, robot
  bring-up) see the X2 deployment repo — this bundle is only the
  fastest possible "watch the policy move" path.
