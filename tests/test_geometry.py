"""Task 6: pixel -> azimuth, the calibrated model, and the calibration maths on synthetic data."""

import math
import sys

import cv2
import numpy as np
import pytest
import yaml

from config import REPO_ROOT
from geometry import AzimuthModel, focal_px, hfov_deg_from_focal, pixel_to_azimuth

sys.path.insert(0, str(REPO_ROOT / "tools"))
import calibrate  # noqa: E402

W, H, HFOV = 1280, 720, 70.4


def test_centre_is_zero_and_edges_are_half_the_fov():
    assert pixel_to_azimuth(W / 2, W, HFOV) == pytest.approx(0.0)
    assert pixel_to_azimuth(0, W, HFOV) == pytest.approx(-HFOV / 2)
    assert pixel_to_azimuth(W, W, HFOV) == pytest.approx(+HFOV / 2)


def test_left_is_negative_monotonic_and_not_linear():
    xs = np.linspace(0, W, 50)
    az = [pixel_to_azimuth(x, W, HFOV) for x in xs]
    assert az[0] < 0 < az[-1] and all(a < b for a, b in zip(az, az[1:]))
    assert pixel_to_azimuth(W / 4, W, HFOV) == pytest.approx(-pixel_to_azimuth(3 * W / 4, W, HFOV))
    assert abs(pixel_to_azimuth(W / 4, W, HFOV)) > HFOV / 4     # a pinhole: more degrees per pixel near the middle


def test_focal_round_trip():
    assert hfov_deg_from_focal(W, focal_px(W, HFOV)) == pytest.approx(HFOV)


def test_model_from_hfov_matches_the_function():
    m = AzimuthModel.from_hfov(W, H, HFOV)
    for x in (0, 100, W / 2, 1000, W):
        assert m.azimuth(x) == pytest.approx(pixel_to_azimuth(x, W, HFOV))
    assert m.hfov_deg == pytest.approx(HFOV)
    assert m.x_at_azimuth(m.azimuth(1000)) == pytest.approx(1000)
    assert m.elevation(W / 2, 0) > 0 > m.elevation(W / 2, H)


def write_cal(path, fx, cx, w, h, dist):
    yaml.safe_dump({"image_width": w, "image_height": h, "dist_coeffs": dist,
                    "camera_matrix": [[fx, 0, cx], [0, fx, h / 2], [0, 0, 1]]}, open(path, "w"))


def test_calibration_file_scales_to_the_frame_size(tmp_path):
    cal = tmp_path / "cam.yaml"
    write_cal(cal, focal_px(W, HFOV), W / 2, W, H, [])
    at_720p = AzimuthModel.from_calibration(cal, W, H)
    at_4k = AzimuthModel.from_calibration(cal, 3840, 2160)
    assert at_4k.azimuth(3840) == pytest.approx(at_720p.azimuth(W)) == pytest.approx(HFOV / 2)
    assert at_4k.azimuth(1000 * 3) == pytest.approx(at_720p.azimuth(1000))
    with pytest.raises(ValueError):                      # 4:3 frame from a 16:9 calibration: different crop
        AzimuthModel.from_calibration(cal, 1280, 960)


def test_barrel_distortion_pushes_edge_angles_outwards(tmp_path):
    cal = tmp_path / "cam.yaml"
    write_cal(cal, focal_px(W, HFOV), W / 2, W, H, [-0.15, 0.02, 0.0, 0.0, 0.0])
    m = AzimuthModel.from_calibration(cal, W, H)
    naive = AzimuthModel.from_hfov(W, H, HFOV)
    assert m.azimuth(W / 2, H / 2) == pytest.approx(0.0, abs=1e-6)
    assert abs(m.azimuth(100, H / 2)) > abs(naive.azimuth(100))   # barrel: the pixel sat closer to centre than the ray


def test_from_config_picks_calibration_when_set(tmp_path):
    cal = tmp_path / "cam.yaml"
    write_cal(cal, 900.0, W / 2, W, H, [])
    assert AzimuthModel.from_config({"hfov_deg": 70.4, "calibration": str(cal)}, W, H).fx == pytest.approx(900.0)
    assert AzimuthModel.from_config({"hfov_deg": 70.4, "calibration": ""}, W, H).hfov_deg == pytest.approx(70.4)


def test_calibration_recovers_known_intrinsics():
    """Project a 9x6 board through a known camera (with distortion) from 15 poses, then calibrate."""
    cols, rows, square = 9, 6, 25.0
    fx, cx, cy = 1000.0, 650.0, 350.0
    k = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
    dist = np.array([-0.12, 0.03, 0.001, -0.001, 0.0])
    obj = calibrate.board_points(cols, rows, square)
    rng = np.random.default_rng(0)
    shots = []
    for _ in range(15):
        rvec = rng.uniform(-0.5, 0.5, 3)
        tvec = np.array([rng.uniform(-120, 20), rng.uniform(-80, 20), rng.uniform(500, 900)])
        pts, _ = cv2.projectPoints(obj, rvec, tvec, k, dist)
        if (pts[:, 0, 0].min() < 0 or pts[:, 0, 0].max() > W or pts[:, 0, 1].min() < 0 or pts[:, 0, 1].max() > H):
            continue
        shots.append(pts.astype(np.float32))
    assert len(shots) >= 8
    cal = calibrate.calibrate(shots, (W, H), cols, rows, square)
    km = np.array(cal["camera_matrix"])
    assert km[0, 0] == pytest.approx(fx, rel=0.01) and km[1, 1] == pytest.approx(fx, rel=0.01)
    assert km[0, 2] == pytest.approx(cx, abs=5) and km[1, 2] == pytest.approx(cy, abs=5)
    assert cal["dist_coeffs"][0] == pytest.approx(dist[0], abs=0.02)
    assert cal["rms_px"] < 0.1
    assert cal["hfov_deg"] == pytest.approx(hfov_deg_from_focal(W, fx), abs=0.5)


def test_find_corners_on_a_drawn_board(tmp_path):
    cols, rows = 9, 6
    calibrate.print_board(tmp_path / "board.png", cols, rows, square_px=40)
    board = cv2.imread(str(tmp_path / "board.png"), cv2.IMREAD_GRAYSCALE)
    assert board.shape == ((rows + 3) * 40, (cols + 3) * 40)
    corners = calibrate.find_corners(board, cols, rows)
    assert corners is not None and corners.shape == (cols * rows, 1, 2)
    xs = np.sort(np.unique(np.round(corners[:, 0, 0] / 40) * 40))
    assert len(xs) == cols and xs[0] == 80 and xs[-1] == 80 + 40 * (cols - 1)   # inner corners sit on the grid
