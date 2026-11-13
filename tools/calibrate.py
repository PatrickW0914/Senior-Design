#!/usr/bin/env python3
"""calibrate.py - camera calibration with a printed checkerboard (Task 6).

  python tools/calibrate.py --print-board board.png            # 1. print this at 100% scale, measure a square
  python tools/calibrate.py --source 0 --square-mm 24.5        # 2. hold the board in front of the webcam
  python tools/calibrate.py --images "data/calibration/shots/*.jpg" --square-mm 24.5
  python tools/calibrate.py --source some_clip.mp4 --square-mm 24.5

Writes data/calibration/<name>.yaml with the camera matrix (focal length, principal point),
distortion coefficients and the horizontal field of view; point camera.calibration in
config.yaml at it. Calibrate at the resolution you run at (config camera.width x height).

Webcam mode shows the live view, draws the corners when the board is found, and captures a
view automatically whenever the board has moved since the last capture, until --views views
are in. Cover the whole frame, especially the corners and edges, and tilt the board. Keys:
space = capture now, c = calibrate with what is captured, q = quit without saving.

--board gives the INNER corners (a board of 10x7 squares has 9x6 inner corners).
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv_env  # noqa: F401, E402
import cv2  # noqa: E402

from config import load_config, repo_path  # noqa: E402
from geometry import hfov_deg_from_focal  # noqa: E402

FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def board_points(cols: int, rows: int, square: float) -> np.ndarray:
    """3-D corner positions on the board (z = 0), in the unit of `square`."""
    pts = np.zeros((rows * cols, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return pts


def find_corners(gray: np.ndarray, cols: int, rows: int) -> np.ndarray | None:
    """Sub-pixel checkerboard corners (N, 1, 2) or None. Searches a downscaled copy first for speed."""
    h, w = gray.shape[:2]
    s = min(1.0, 800 / w)
    small = cv2.resize(gray, None, fx=s, fy=s) if s < 1 else gray
    ok, corners = cv2.findChessboardCorners(small, (cols, rows), flags=FLAGS)
    if not ok:
        return None
    corners = (corners / s).astype(np.float32).reshape(-1, 1, 2)   # OpenCV 5 returns (N, 2), 4 gave (N, 1, 2)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX).reshape(-1, 1, 2)


def calibrate(image_points: list[np.ndarray], size: tuple[int, int], cols: int, rows: int, square: float) -> dict:
    """OpenCV calibration from corner sets. Returns a dict ready to write as yaml."""
    obj = [board_points(cols, rows, square)] * len(image_points)
    rms, k, dist, _, _ = cv2.calibrateCamera(obj, image_points, size, None, None)
    w, h = size
    return {
        "image_width": int(w), "image_height": int(h),
        "camera_matrix": [[float(v) for v in row] for row in k],
        "dist_coeffs": [float(v) for v in dist.ravel()],
        "rms_px": float(rms), "n_views": len(image_points),
        "hfov_deg": float(hfov_deg_from_focal(w, k[0, 0])),
        "vfov_deg": float(hfov_deg_from_focal(h, k[1, 1])),
        "board": {"cols": cols, "rows": rows, "square_mm": square},
        "date": str(date.today()),
    }


def print_board(path: Path, cols: int, rows: int, square_px: int = 200) -> None:
    """A checkerboard image with (cols+1) x (rows+1) squares, plus a white margin."""
    sq_cols, sq_rows, m = cols + 1, rows + 1, square_px
    img = np.full(((sq_rows + 2) * m, (sq_cols + 2) * m), 255, np.uint8)
    for r in range(sq_rows):
        for c in range(sq_cols):
            if (r + c) % 2 == 0:
                img[(r + 1) * m:(r + 2) * m, (c + 1) * m:(c + 2) * m] = 0
    cv2.imwrite(str(path), img)
    print(f"wrote {path}: {sq_cols}x{sq_rows} squares = {cols}x{rows} inner corners. "
          "Print at 100% on stiff paper, then measure one square and pass --square-mm.")


def moved_enough(corners: np.ndarray, last: np.ndarray | None, min_px: float) -> bool:
    return last is None or float(np.linalg.norm(corners.mean(0) - last.mean(0))) > min_px


def collect_from_webcam(index: int, cam_cfg: dict, cols: int, rows: int, views: int):
    cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {index}")
    shots, last, last_t, size = [], None, 0.0, None
    win = "calibrate (space capture, c calibrate, q quit)"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            size = (frame.shape[1], frame.shape[0])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = find_corners(gray, cols, rows)
            key = cv2.waitKey(1) & 0xFF
            auto = corners is not None and time.time() - last_t > 0.7 and moved_enough(corners, last, 0.08 * size[0])
            if corners is not None and (auto or key == ord(" ")):
                shots.append(corners)
                last, last_t = corners, time.time()
                print(f"captured view {len(shots)}/{views}")
            view = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(view, (cols, rows), corners, True)
            cv2.putText(view, f"views {len(shots)}/{views}   board {'FOUND' if corners is not None else 'not found'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if corners is not None else (0, 0, 255), 2)
            cv2.imshow(win, view)
            if key == ord("q"):
                shots = []
                break
            if key == ord("c") or len(shots) >= views:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return shots, size, f"webcam {index}"


def collect_from_frames(frames, cols: int, rows: int, views: int, label: str):
    shots, last, size = [], None, None
    for frame in frames:
        size = (frame.shape[1], frame.shape[0])
        corners = find_corners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cols, rows)
        if corners is not None and moved_enough(corners, last, 0.05 * size[0]):
            shots.append(corners)
            last = corners
            if len(shots) >= views:
                break
    return shots, size, label


def video_frames(path: str, step: int):
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield frame
        i += 1
    cap.release()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", help="camera index or video file")
    p.add_argument("--images", help="glob of still images, e.g. \"shots/*.jpg\"")
    p.add_argument("--board", default="9x6", help="inner corners, COLSxROWS (default 9x6)")
    p.add_argument("--square-mm", type=float, default=25.0, help="measured square size in mm (default 25)")
    p.add_argument("--views", type=int, default=20, help="views to collect (default 20)")
    p.add_argument("--step", type=int, default=10, help="video: try every Nth frame (default 10)")
    p.add_argument("--out", help="output yaml (default data/calibration/<source>_<WxH>.yaml)")
    p.add_argument("--print-board", metavar="PNG", help="write a printable checkerboard and exit")
    args = p.parse_args(argv)
    cols, rows = (int(v) for v in args.board.lower().split("x"))
    if args.print_board:
        print_board(Path(args.print_board), cols, rows)
        return 0
    if not args.source and not args.images:
        p.error("give --source or --images (or --print-board)")

    cfg = load_config()
    if args.images:
        paths = sorted(glob.glob(args.images))
        shots, size, label = collect_from_frames((cv2.imread(f) for f in paths), cols, rows, args.views, "images")
        name = "images"
    elif str(args.source).isdigit():
        shots, size, label = collect_from_webcam(int(args.source), cfg["camera"], cols, rows, args.views)
        name = f"camera{args.source}"
    else:
        shots, size, label = collect_from_frames(video_frames(args.source, args.step), cols, rows, args.views, args.source)
        name = Path(args.source).stem
    if len(shots) < 8:
        print(f"only {len(shots)} usable views (need 8 or more, 20 is better): nothing written", file=sys.stderr)
        return 1
    cal = calibrate(shots, size, cols, rows, args.square_mm)
    cal["source"] = label
    out = Path(args.out) if args.out else repo_path("data/calibration") / f"{name}_{size[0]}x{size[1]}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cal, fh, sort_keys=False)
    k = cal["camera_matrix"]
    print(f"{len(shots)} views, rms {cal['rms_px']:.3f} px ({'good' if cal['rms_px'] < 0.5 else 'high: recapture with a flat board and more tilt'})")
    print(f"fx {k[0][0]:.1f}  fy {k[1][1]:.1f}  cx {k[0][2]:.1f}  cy {k[1][2]:.1f}  "
          f"hfov {cal['hfov_deg']:.1f} deg  vfov {cal['vfov_deg']:.1f} deg  dist {np.round(cal['dist_coeffs'], 4).tolist()}")
    print(f"wrote {out}\nset in config.yaml:  camera.calibration: {out.relative_to(repo_path('.')) if out.is_relative_to(repo_path('.')) else out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
