#!/usr/bin/env bash
# Idle stand (hands-on-back loop) in MuJoCo with the SONIC policy.
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "No .venv — run ./install.sh first." >&2; exit 1; }
exec .venv/bin/python scripts/eval_x2_mujoco_onnx.py \
    --onnx "${MODEL:-models/x2_sonic_14000_g1.onnx}" \
    --motion motions/x2_idle_stand.pkl \
    "$@"
