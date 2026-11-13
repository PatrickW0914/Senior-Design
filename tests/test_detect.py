"""Task 4: box mapping, frontal check, and (when models and a clip are present) a real detection."""

import cv2
import numpy as np
import pytest

from config import REPO_ROOT, load_config, repo_path
from detect import Detection, FaceDetector, is_frontal, scale_detection, yaw_ratio


def test_yaw_ratio_and_frontal():
    frontal = np.array([[40.0, 50.0], [60.0, 50.0], [50.0, 65.0]])      # nose under the eye midpoint
    turned = np.array([[40.0, 50.0], [60.0, 50.0], [62.0, 65.0]])       # nose past one eye
    assert yaw_ratio(frontal) == pytest.approx(0.0)
    assert yaw_ratio(turned) == pytest.approx(0.6)
    assert is_frontal(Detection((0, 0, 1, 1), 1.0, frontal))
    assert not is_frontal(Detection((0, 0, 1, 1), 1.0, turned))
    assert yaw_ratio(None) is None and not is_frontal(Detection((0, 0, 1, 1), 1.0, None))


def test_scale_detection_maps_back_and_clips():
    d = Detection((10.0, 20.0, 30.0, 60.0), 0.9, np.array([[12.0, 25.0], [28.0, 25.0], [20.0, 40.0]]))
    full = scale_detection(d, 4.0, frame_w=100, frame_h=200)
    assert full.bbox == (40.0, 80.0, 100.0, 200.0)                       # x2, y2 clipped to the frame
    assert full.keypoints.tolist() == [[48.0, 100.0], [112.0, 100.0], [80.0, 160.0]]
    assert full.score == 0.9 and full.height == 120.0 and full.width == 60.0 and full.cx == 70.0


def test_detector_downscales_and_sorts_largest_first():
    seen = {}

    def fake_impl(img):
        seen["shape"] = img.shape
        return [Detection((10, 10, 20, 30), 0.8), Detection((50, 10, 90, 90), 0.7)]   # small face first

    det = FaceDetector({"backend": "scrfd", "width": 100}, _impl=fake_impl)
    out = det.detect(np.zeros((300, 400, 3), np.uint8))                 # 400 wide -> 100 wide, factor 4
    assert seen["shape"] == (75, 100, 3)
    assert [d.bbox for d in out] == [(200.0, 40.0, 360.0, 300.0), (40.0, 40.0, 80.0, 120.0)]   # y2 clipped to 300
    assert det.last_ms >= 0.0
    # a frame narrower than detect.width is used as is
    out = FaceDetector({"width": 1000}, _impl=fake_impl).detect(np.zeros((300, 400, 3), np.uint8))
    assert seen["shape"] == (300, 400, 3) and out[0].bbox == (50.0, 10.0, 90.0, 90.0)


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        FaceDetector({"backend": "yolo"})


CLIP = REPO_ROOT / "data" / "videos" / "two_person_podcast.mp4"
_cfg = load_config()["detect"]
_pack_dir = repo_path(_cfg["scrfd"]["root"]) / "models" / _cfg["scrfd"]["pack"]


@pytest.mark.skipif(not CLIP.is_file() or not _pack_dir.is_dir(), reason="needs the podcast clip and the SCRFD pack")
def test_scrfd_finds_the_podcast_host():
    cap = cv2.VideoCapture(str(CLIP))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(30 * cap.get(cv2.CAP_PROP_FPS)))
    ok, frame = cap.read()
    cap.release()
    assert ok
    det = FaceDetector(_cfg)
    faces = det.detect(frame)
    assert len(faces) == 1
    f = faces[0]
    assert f.height > 150 and f.score > 0.5 and f.keypoints.shape == (5, 2)
    assert 0 <= f.bbox[0] < f.bbox[2] <= frame.shape[1] and 0 <= f.bbox[1] < f.bbox[3] <= frame.shape[0]
    assert is_frontal(f)
    assert det.last_ms < 200
