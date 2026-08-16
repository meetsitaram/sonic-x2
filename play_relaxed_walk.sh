#!/usr/bin/env bash
# relaxed_walk in MuJoCo with the SONIC policy.
# DEFAULT MODEL: frozen-g1core + LoRA transfer (v2) — the model from
# https://sonic-agibot-x2.github.io/sonic-transfer/ — with its required
# runtime config (parity gains + action clip + frozen wrists).
# Incumbent: MODEL=models/x2_sonic_14000_g1.onnx ./play_relaxed_walk.sh
# (the incumbent auto-uses its bigrun deploy-tuning preset instead).
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "No .venv — run ./install.sh first." >&2; exit 1; }
MODEL="${MODEL:-models/x2_sonic_frozen_g1core_lora_v2.onnx}"
EXTRA=()
case "$MODEL" in
  *frozen_g1core*) EXTRA=(--tuning '' --action-clip 20 --freeze-wrist) ;;
esac
exec .venv/bin/python scripts/eval_x2_mujoco_onnx.py \
    --onnx "$MODEL" \
    --motion motions/x2_relaxed_walk.pkl \
    "${EXTRA[@]}" \
    "$@"
