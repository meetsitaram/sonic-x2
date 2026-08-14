#!/usr/bin/env bash
# Create a local venv with everything needed to run the SONIC X2 MuJoCo
# player (CPU-only: mujoco + onnxruntime, no torch).
#
# Requires: python3.10+ with the venv module, internet access.
# Usage:    ./install.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
echo "Using interpreter: $($PY --version)"

$PY -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install \
    "mujoco>=3.1" \
    "onnxruntime>=1.17" \
    "numpy<2" \
    scipy \
    joblib \
    pyyaml

# ---- X2 vendor meshes (AgiBot official URDF package; not redistributed
# in this repo — see assets/urdf/x2_ultra/meshes/README.md) ----
X2_URDF_URL="${X2_URDF_URL:-https://x2-aimdk.agibot.com/en/latest/_downloads/2ffc9785259556f409e385974a7a0461/X2_URDF-v1.3.0.zip}"
MESH_DEST="assets/urdf/x2_ultra/meshes"
if compgen -G "${MESH_DEST}/*.STL" > /dev/null; then
    echo "X2 meshes already present — skipping download"
else
    echo "Fetching AgiBot X2 URDF package (~50 MB; X2_URDF_URL to override)..."
    _tmp="$(mktemp -d)"
    curl -fSL --retry 2 -o "${_tmp}/x2_urdf.zip" "${X2_URDF_URL}"
    (cd "${_tmp}" && unzip -q x2_urdf.zip "*/meshes/*")
    find "${_tmp}" -type f \( -name "*.STL" -o -name "*.stl" \) -exec cp {} "${MESH_DEST}/" \;
    rm -rf "${_tmp}"
    echo "Installed $(ls ${MESH_DEST} | grep -ci stl) meshes."
fi

echo
echo "Install OK. Smoke-checking imports..."
.venv/bin/python - <<'EOF'
import mujoco, onnxruntime, numpy, scipy, joblib
print("mujoco", mujoco.__version__, "| onnxruntime", onnxruntime.__version__,
      "| numpy", numpy.__version__)
model = mujoco.MjModel.from_xml_path("assets/mjcf/x2_ultra.xml")
print("X2 Ultra MJCF loads:", model.njnt, "joints,", model.nu, "actuators")
EOF

echo
echo "Done. Try:  ./play_relaxed_walk.sh   ./play_idle_stand.sh   ./play_gangam_dance.sh"
