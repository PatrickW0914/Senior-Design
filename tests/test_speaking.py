"""Task 8: mouth ratio, its fluctuation, the probability curve and the scorer's bookkeeping."""

import math

import cv2
import numpy as np
import pytest

from config import REPO_ROOT, load_config, repo_path
from speaking import LipMotion, SpeakingScorer, head_pose, opening_ratio, probability, variability
from track import Track

FPS = 30


def test_opening_ratio_is_scale_and_rotation_free():
    upper, lower, left, right = [0, -5, 0], [0, 5, 0], [-25, 0, 0], [25, 0, 0]
    assert opening_ratio(upper, lower, left, right) == pytest.approx(0.2)
    big = [np.array(p) * 7.5 for p in (upper, lower, left, right)]                     # closer to the camera
    assert opening_ratio(*big) == pytest.approx(0.2)
    yaw = math.radians(60)                                                             # head turned 60 degrees
    rot = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    turned = [rot @ np.array(p, dtype=float) for p in (upper, lower, left, right)]
    assert opening_ratio(*turned) == pytest.approx(0.2)
    flat = [p[:2] for p in turned]                                                     # what 2-D alone would say
    assert opening_ratio(*flat) == pytest.approx(0.4)                                  # twice as open: wrong
    assert opening_ratio(upper, lower, left, left) is None


def series(f, seconds=0.75, fps=FPS):
    ts = np.arange(int(seconds * fps)) / fps
    return ts, np.array([f(t) for t in ts])


@pytest.mark.parametrize("method", ["std", "band"])
def test_variability_separates_speech_from_steady_and_slow(method):
    amp = 0.05
    talk = variability(*series(lambda t: 0.1 + amp * math.sin(2 * math.pi * 5 * t)), method)
    still = variability(*series(lambda t: 0.02), method)
    smile = variability(*series(lambda t: 0.3), method)                                # wide but steady
    drift = variability(*series(lambda t: 0.05 + 0.08 * t), method)                    # slowly opening
    assert still == pytest.approx(0, abs=1e-9) and smile == pytest.approx(0, abs=1e-9)
    assert talk > 0.5 * amp and drift < 0.2 * talk
    assert variability([0.0, 0.1], [0.1, 0.2], method) == 0.0                          # too few samples


def test_std_is_the_deviation_and_unknown_method_is_refused():
    ts, rs = series(lambda t: 0.1 + 0.05 * math.sin(2 * math.pi * 4 * t), seconds=1.0)
    assert variability(ts, rs, "std") == pytest.approx(0.05 / math.sqrt(2), rel=0.1)
    with pytest.raises(ValueError):
        variability(ts, rs, "wavelet")


def test_probability_curve():
    assert probability(0.02, 0.02, 0.005) == pytest.approx(0.5)
    assert probability(0.0, 0.02, 0.005) < 0.05 and probability(0.06, 0.02, 0.005) > 0.99
    ps = [probability(v, 0.02, 0.005) for v in np.linspace(0, 0.1, 30)]
    assert all(a <= b for a, b in zip(ps, ps[1:]))
    assert probability(1e9, 0.02, 0.005) == pytest.approx(1.0)                         # no overflow


def feed(motion, f, t0, seconds):
    out = []
    for i in range(int(seconds * FPS)):
        t = t0 + i / FPS
        out.append(motion.update(t, f(t)))
    return out


def test_lip_motion_talking_silent_and_smile():
    m = LipMotion(window_s=0.75, threshold=0.02, slope=0.005)
    early = feed(m, lambda t: 0.1 + 0.05 * math.sin(2 * math.pi * 5 * t), 0.0, 0.3)
    assert max(early) == 0.0                                                           # window not filled yet
    talking = feed(m, lambda t: 0.1 + 0.05 * math.sin(2 * math.pi * 5 * t), 0.3, 1.0)
    assert talking[-1] > 0.9
    silent = feed(m, lambda t: 0.02, 1.3, 1.5)
    assert silent[-1] < 0.05
    smile = feed(m, lambda t: 0.30, 2.8, 1.5)                                          # jumps wide, then holds
    assert max(smile[:5]) > 0.5 and smile[-1] < 0.05                                   # the jump shows, the hold does not


def test_lip_motion_hold_and_time_reset():
    m = LipMotion(window_s=0.75, threshold=0.02, slope=0.005, hold_s=0.3)
    p = feed(m, lambda t: 0.1 + 0.05 * math.sin(2 * math.pi * 5 * t), 0.0, 1.5)[-1]
    last = 1.5 - 1 / FPS
    assert m.hold(last + 0.2) == pytest.approx(p)                                      # landmarks lost briefly
    assert m.hold(last + 0.5) == 0.0                                                   # ... for too long
    assert m.update(0.0, 0.1) == 0.0 and len(m._hist) == 1                             # a new clip starts at 0


def test_scorer_bookkeeping_with_scripted_ratios(monkeypatch):
    sc = SpeakingScorer({"window_s": 0.75, "threshold": 0.02, "slope": 0.005, "hold_s": 0.3})
    frame = np.zeros((48, 64, 3), np.uint8)
    a, b = Track(1, (0, 0, 10, 10), 0.9), Track(2, (20, 0, 30, 10), 0.9)
    ratios = {}
    monkeypatch.setattr(sc, "mouth_ratio", lambda track, frame: ratios[track.track_id])
    for i in range(45):
        t = i / FPS
        ratios = {1: 0.1 + 0.05 * math.sin(2 * math.pi * 5 * t), 2: 0.02}
        pa, pb = sc.score(a, frame, t), sc.score(b, frame, t)
    assert pa > 0.9 and pb < 0.05
    assert sc.mouth[1] == pytest.approx(ratios[1]) and sc.mouth[2] == 0.02
    assert sc.variability[1] > 0.02 > sc.variability[2]
    ratios = {1: None, 2: 0.02}                                                        # #1's landmarks drop out
    assert sc.score(a, frame, 1.5) == pytest.approx(pa) and 1 not in sc.mouth
    assert sc.score(a, frame, 2.5) == 0.0
    sc.score(b, frame, 9.0)                                                            # #1 unseen for > 5 s
    assert 1 not in sc._motion and 1 not in sc.variability and 2 in sc._motion


def test_missing_model_says_how_to_get_it(tmp_path):
    sc = SpeakingScorer({"model_path": str(tmp_path / "nope.task")})
    with pytest.raises(FileNotFoundError, match="get_models"):
        sc.mouth_ratio(Track(1, (10, 10, 40, 40), 0.9), np.zeros((48, 64, 3), np.uint8))


CLIP = REPO_ROOT / "data" / "videos" / "two_person_podcast.mp4"
_cfg = load_config()
_have = (CLIP.is_file() and repo_path(_cfg["speaking"]["model_path"]).is_file()
         and (repo_path(_cfg["detect"]["scrfd"]["root"]) / "models" / _cfg["detect"]["scrfd"]["pack"]).is_dir())


@pytest.mark.skipif(not _have, reason="needs the podcast clip and the model files")
def test_real_landmarks_give_a_plausible_ratio():
    from detect import FaceDetector

    cap = cv2.VideoCapture(str(CLIP))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(30 * cap.get(cv2.CAP_PROP_FPS)))
    ok, frame = cap.read()
    cap.release()
    assert ok
    face = FaceDetector(_cfg["detect"]).detect(frame)[0]
    ratio = SpeakingScorer(_cfg["speaking"]).mouth_ratio(Track(1, face.bbox, face.score), frame)
    assert ratio is not None and 0.0 <= ratio < 1.0


def test_head_pose_from_keypoints():
    frontal = np.array([[40.0, 50.0], [60.0, 50.0], [50.0, 62.0], [44.0, 75.0], [56.0, 75.0]])
    assert head_pose(frontal) == pytest.approx((0.0, 0.6))
    profile = np.array([[52.0, 50.0], [60.0, 50.0], [75.0, 62.0]])                     # eyes close together, nose far out
    assert head_pose(profile)[0] == pytest.approx(2.375)
    assert head_pose(None) is None and head_pose(np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])) is None


def test_lip_motion_needs_unbroken_landmarks():
    talk = lambda t: 0.1 + 0.05 * math.sin(2 * math.pi * 5 * t)                         # noqa: E731
    m = LipMotion(max_gap_s=0.2)
    for i in range(30):
        m.update(i / FPS, talk(i / FPS))
    assert m.prob > 0.9
    scattered = LipMotion(max_gap_s=0.2)                                               # a landmark every 0.3 s
    probs = [scattered.update(i * 0.3, talk(i * 0.3)) for i in range(12)]
    assert max(probs) == 0.0
    lenient = LipMotion(window_s=2.0, max_gap_s=None)                                  # the same samples, gaps allowed
    assert max(lenient.update(i * 0.3, 0.1 + 0.1 * (i % 2)) for i in range(12)) > 0.9


def test_lip_motion_distrusts_a_moving_head():
    talk = lambda t: 0.1 + 0.05 * math.sin(2 * math.pi * 5 * t)                         # noqa: E731
    still = LipMotion(max_head_motion=0.1)
    turning = LipMotion(max_head_motion=0.1)
    for i in range(30):
        t = i / FPS
        still.update(t, talk(t), yaw=0.05, pitch=0.6)
        turning.update(t, talk(t), yaw=0.05 + 0.7 * t, pitch=0.6)                       # the head swings 40 degrees
    assert still.prob > 0.9 and still.head_motion < 0.01
    assert turning.prob == 0.0 and turning.head_motion > 0.1 and turning.variability > 0.02
    no_pose = LipMotion(max_head_motion=0.1)                                           # no keypoints: nothing to gate on
    for i in range(30):
        no_pose.update(i / FPS, talk(i / FPS))
    assert no_pose.prob > 0.9


def test_scorer_does_not_score_a_turned_face(monkeypatch):
    sc = SpeakingScorer({"max_yaw": 0.6})
    looked = []
    monkeypatch.setattr(sc, "mouth_ratio", lambda track, frame: looked.append(track.track_id) or 0.1)
    frame = np.zeros((48, 64, 3), np.uint8)
    profile = Track(1, (0, 0, 10, 10), 0.9, np.array([[52.0, 50.0], [60.0, 50.0], [75.0, 62.0]]))
    frontal = Track(2, (0, 0, 10, 10), 0.9, np.array([[40.0, 50.0], [60.0, 50.0], [50.0, 62.0]]))
    assert sc.score(profile, frame, 0.0) == 0.0 and looked == [] and 1 not in sc.mouth   # the landmarker is not even run
    sc.score(frontal, frame, 0.0)
    assert looked == [2] and sc.mouth[2] == 0.1
    assert sc.measure(profile, frame) == (0.1, pytest.approx(2.375), pytest.approx(1.5))  # ungated, for the eval cache
