"""Task 8: the evaluation's bookkeeping - replay, truth sources, person mapping, metrics."""

import math
import sys

import pytest

from config import REPO_ROOT
from labels import Labels, Segment

sys.path.insert(0, str(REPO_ROOT / "eval"))
import speaking_accuracy as ev  # noqa: E402

SP = {"window_s": 0.75, "threshold": 0.02, "slope": 0.005, "hold_s": 0.3, "method": "std",
      "on_threshold": 0.6, "off_threshold": 0.4}
FPS = 30


def two_people(seconds=4.0):
    """Left face (track 5, cx 300) talks for the first half, right face (track 9, cx 900) for the second."""
    frames = []
    for i in range(int(seconds * FPS)):
        t = i / FPS
        wiggle = 0.1 + 0.05 * math.sin(2 * math.pi * 5 * t)
        frames.append((t, [(5, 300.0, wiggle if t < seconds / 2 else 0.02),
                           (9, 900.0, 0.02 if t < seconds / 2 else wiggle)]))
    return frames


def test_replay_and_decide_follow_who_is_talking():
    rows = ev.replay(two_people(), SP)
    prob, pred = ev.decide(rows, SP)
    at = {(round(t, 3), tid): p for (t, tid, *_), p in zip(rows, pred)}
    assert at[(1.5, 5)] and not at[(1.5, 9)]
    assert at[(3.5, 9)] and not at[(3.5, 5)]
    assert len(rows) == 2 * 120 and all(0.0 <= p <= 1.0 for p in prob)


def test_person_mapping_and_label_truth(tmp_path):
    rows = ev.replay(two_people(), SP)
    assert ev.person_of_track(rows, 2) == {5: 1, 9: 2}             # left to right
    assert ev.person_of_track(rows, 3) == {}                       # never three faces in a frame
    visitor = rows + [(1.0, 77, 620.0, 0.02, 0.0, 200.0)]          # someone standing between the two seats
    assert ev.person_of_track(visitor, 2, width=1280)  == {5: 1, 9: 2}
    assert ev.person_of_track(visitor, 2)[77] in (1, 2)            # without a width everyone is assigned
    Labels([Segment(0.0, 2.0, 1), Segment(2.0, 4.0, 2)]).save(tmp_path / "clip.csv")
    source, truth, any_face = ev.truth_for(tmp_path / "clip.mp4", rows, tmp_path)
    assert source == "labels" and not any_face
    _, pred = ev.decide(rows, SP)
    s = ev.scores(*ev.confusion(pred, truth, rows, any_face))
    assert s["n"] == 240 and s["accuracy"] > 0.8                   # the 0.75 s window lags each change a little
    assert s["precision"] > 0.8 and s["recall"] > 0.75


def test_header_only_label_file_means_nobody_speaks(tmp_path):
    Labels().save(tmp_path / "quiet.csv")
    rows = ev.replay(two_people(), SP)
    source, truth, _ = ev.truth_for(tmp_path / "quiet.mp4", rows, tmp_path)
    assert source == "labels (no speech)" and truth == [False] * len(rows)
    s = ev.scores(*ev.confusion(ev.decide(rows, SP)[1], truth, rows, False))
    assert s["false_positive_rate"] == pytest.approx(1 - s["accuracy"]) and math.isnan(s["recall"])


def test_no_labels_and_no_clip_is_unjudged(tmp_path):
    assert ev.truth_for(tmp_path / "missing.mp4", [], tmp_path) == ("none", None, False)


def test_confusion_per_face_and_any_face():
    rows = [(0.0, 1, 0, 0, 0), (0.0, 2, 0, 0, 0), (1.0, 1, 0, 0, 0), (1.0, 2, 0, 0, 0)]
    pred = [True, False, False, False]
    assert ev.confusion(pred, [True, False, True, None], rows, False) == (1, 0, 1, 1)      # None is not judged
    assert ev.confusion(pred, [True, True, True, True], rows, True) == (1, 0, 1, 0)        # one decision per frame


def test_auc():
    assert ev.auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert ev.auc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    assert ev.auc([0.5, 0.5, 0.5, 0.5], [False, True, False, True]) == pytest.approx(0.5)
    assert math.isnan(ev.auc([0.1, 0.2], [True, True]))


def test_load_rows_round_trip(tmp_path):
    path = tmp_path / "c.csv"
    path.write_text("t,track_id,cx,cy,h,ratio,frame_w,frame_h\n0.0000,,,,,,1280,720\n"
                    "0.0333,3,640.0,300.0,200.0,0.12345,1280,720\n0.0667,3,641.0,300.0,200.0,,1280,720\n")
    frames, width = ev.load_rows(path)
    assert width == 1280 and [t for t, _ in frames] == [0.0, 0.0333, 0.0667]
    assert frames[0][1] == [] and frames[1][1] == [(3, 640.0, 0.12345, 300.0, None, None)]      # an older cache: no pose
    assert frames[2][1] == [(3, 641.0, None, 300.0, None, None)]


def test_face_labels_are_matched_by_box(tmp_path):
    """Imported datasets: each tracked face takes the truth of the labeled box it is in."""
    from labels import FaceLabel, FaceLabels

    frames = [(t, [(5, 300.0, r5, 200.0), (9, 900.0, r9, 220.0)]) for t, faces in two_people()
              for (_, _, r5), (_, _, r9) in [faces]]
    rows = ev.replay(frames, SP)
    labels = []
    for t, _ in frames:
        labels.append(FaceLabel(t, 250, 150, 350, 250, t < 2.0, 1, "face"))        # around the left face
        labels.append(FaceLabel(t, 780, 150, 1020, 600, t >= 2.0, 2, "body"))      # upper body of the right one
    FaceLabels(labels).save(tmp_path / "clip.faces.csv")
    source, truth, any_face = ev.truth_for(tmp_path / "clip.mp4", rows, tmp_path)
    assert source == "face labels" and not any_face and None not in truth
    s = ev.scores(*ev.confusion(ev.decide(rows, SP)[1], truth, rows, any_face))
    assert s["n"] == 240 and s["accuracy"] > 0.8
    far = ev.replay([(0.0, [(7, 600.0, 0.02, 650.0)])], SP)                        # a face in no labeled box
    assert ev.truth_for(tmp_path / "clip.mp4", far, tmp_path)[1] == [None]


def test_replay_applies_the_pose_gates():
    """A face turned past max_yaw, or turning during the window, gets no probability."""
    def clip(yaw_of):
        return [(i / FPS, [(1, 300.0, 0.1 + 0.05 * math.sin(2 * math.pi * 5 * i / FPS), 200.0, yaw_of(i / FPS), 0.5)])
                for i in range(90)]
    talking_frontal = ev.replay(clip(lambda t: 0.05), {**SP, "max_yaw": 0.6, "max_head_motion": 0.1})
    assert ev.decide(talking_frontal, SP)[1][-1] and talking_frontal[-1][6]
    profile = ev.replay(clip(lambda t: 2.3), {**SP, "max_yaw": 0.6})
    assert not any(ev.decide(profile, SP)[1]) and not profile[-1][6]
    assert any(ev.decide(ev.replay(clip(lambda t: 2.3), SP), SP)[1])             # without the gate it would fire
    turning = ev.replay(clip(lambda t: t), {**SP, "max_head_motion": 0.1})        # yaw sweeping 0 -> 3
    assert not any(ev.decide(turning, SP)[1])
