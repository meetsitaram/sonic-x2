#!/usr/bin/env bash
# Create a local venv with everything needed to run the SONIC X2 MuJoCo
# player. Torch is required by the player (CPU wheel is enough — the
# policy itself runs through onnxruntime).
#
# Requires: python3.10+ with the venv module, internet access.
# Usage:    ./install.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
echo "Using interpreter: $($PY --version)"

$PY -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install \
    "mujoco>=3.1" \
    "onnxruntime>=1.17" \
    "numpy<2" \
    scipy \
    joblib \
    pyyaml

echo
echo "Install OK. Smoke-checking imports..."
.venv/bin/python - <<'EOF'
import mujoco, onnxruntime, torch, numpy, scipy, joblib
print("mujoco", mujoco.__version__, "| onnxruntime", onnxruntime.__version__,
      "| torch", torch.__version__, "| numpy", numpy.__version__)
model = mujoco.MjModel.from_xml_path("assets/mjcf/x2_ultra.xml")
print("X2 Ultra MJCF loads:", model.njnt, "joints,", model.nu, "actuators")
EOF

echo
echo "Done. Try:  ./play_relaxed_walk.sh   ./play_idle_stand.sh   ./play_gangam_dance.sh"
