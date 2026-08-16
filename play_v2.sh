#!/usr/bin/env bash
# Run the frozen-G1core + LoRA transfer model (v2) in MuJoCo.
# This is the model from https://sonic-agibot-x2.github.io/sonic-transfer/ —
# it beats the native incumbent OOD (PHUMA 69.0 vs 59.0).
#
# Usage:  ./play_v2.sh [gangam|walk|idle]     (default: gangam)
#
# v2 REQUIRES its own runtime config (do NOT use the incumbent's bigrun
# tuning preset — its deviation clamps destabilize this model):
#   parity gains, action clip 20, wrists frozen (deploy-default).
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "No .venv — run ./install.sh first." >&2; exit 1; }
CLIP="${1:-gangam}"
case "$CLIP" in
  gangam) MOTION=motions/x2_gangam_dance.pkl ;;
  walk)   MOTION=motions/x2_relaxed_walk.pkl ;;
  idle)   MOTION=motions/x2_idle_stand.pkl ;;
  *) echo "unknown clip '$CLIP' (gangam|walk|idle)" >&2; exit 2 ;;
esac
shift || true
exec .venv/bin/python scripts/eval_x2_mujoco_onnx.py \
    --onnx models/x2_sonic_frozen_g1core_lora_v2.onnx \
    --motion "$MOTION" \
    --tuning '' --action-clip 20 --freeze-wrist \
    "$@"
