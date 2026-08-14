#!/usr/bin/env bash
# Gangnam-style victory dance in MuJoCo with the SONIC policy — one of the
# dance-bank clips bound to the robot's gamepad demo. Uses the bigrun demo
# tuning preset (the gantry-verified config the dances deploy with).
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "No .venv — run ./install.sh first." >&2; exit 1; }
exec .venv/bin/python scripts/eval_x2_mujoco_onnx.py \
    --onnx models/x2_sonic_14000_g1.onnx \
    --motion motions/x2_gangam_dance.pkl \
    --tuning configs/real_deploy_tuning/bigrun.yaml \
    "$@"
