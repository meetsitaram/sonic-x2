#!/usr/bin/env python3
"""Offscreen kinematic playback of a motion PKL to mp4 (no physics, no policy).

Poses the X2 model directly from the clip frames (PKL ``dof`` is in MJCF
qpos convention) and pipes the render to ffmpeg at the clip's fps. One pass
over the selected clip(s), then exits.

Usage:
    .venv/bin/python scripts/record_kinematic.py \
        --motion motions/x2_relaxed_walk.pkl --out media/walk_kinematic.mp4
"""
import argparse
import sys
from pathlib import Path

import joblib
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_x2_mujoco import MJCF_PATH, NUM_DOFS  # noqa: E402
from eval_x2_mujoco_onnx import VideoRecorder  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion", required=True, help="Reference motion PKL.")
    ap.add_argument("--clip", default=None,
                    help="Exact key, or substring filter (default: all clips).")
    ap.add_argument("--out", required=True, help="Output mp4 path.")
    ap.add_argument("--init-frame", type=int, default=0)
    args = ap.parse_args()

    data = joblib.load(args.motion)
    names = list(data.keys())
    if args.clip:
        if args.clip in names:
            names = [args.clip]
        else:
            names = [k for k in names if args.clip.lower() in k.lower()]
        if not names:
            raise SystemExit(f"--clip '{args.clip}' matched nothing")

    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    mjd = mujoco.MjData(model)
    fps = float(data[names[0]]["fps"])
    rec = VideoRecorder(args.out, model, model.body("pelvis").id, fps)
    for name in names:
        m = data[name]
        dof = np.asarray(m["dof"])
        print(f"[kinematic] {name}: {dof.shape[0]} frames @ {m['fps']:g} fps")
        for f in range(args.init_frame, dof.shape[0]):
            mjd.qpos[0:3] = m["root_trans_offset"][f]
            q = np.asarray(m["root_rot"][f])  # xyzw
            mjd.qpos[3:7] = [q[3], q[0], q[1], q[2]]
            mjd.qpos[7:7 + NUM_DOFS] = dof[f]
            mjd.qvel[:] = 0
            mujoco.mj_forward(model, mjd)
            rec.capture(mjd)
    rec.close()


if __name__ == "__main__":
    main()
