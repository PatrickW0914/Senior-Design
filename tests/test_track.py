"""Task 5: track IDs on synthetic detections - stable, crossing, surviving gaps, expiring."""

import numpy as np
import pytest

from detect import Detection
from track import FaceTracker, Track

FPS = 30
CFG = {"max_lost_s": 2.0, "activation_threshold": 0.5, "match_threshold": 0.8, "min_consecutive_frames": 1,
       "lost_velocity_decay": 0.8, "reacquire_radius_h": 1.5}
FRAME = np.zeros((360, 640, 3), np.uint8)


def box(x, y=100, w=60, h=80, score=0.9, kps=None):
    return Detection((float(x), float(y), float(x + w), float(y + h)), score, kps)


def run(tracker, frames):
    """frames: list of detection lists. Returns the list of {track_id: Track} per frame."""
    return [{tr.track_id: tr for tr in tracker.update(dets, FRAME, i / FPS)} for i, dets in enumerate(frames)]


def test_two_separated_faces_keep_their_ids():
    out = run(FaceTracker(CFG, FPS), [[box(10 + i), box(400)] for i in range(30)])
    ids_left = {min(f.values(), key=lambda tr: tr.cx).track_id for f in out[1:]}
    ids_right = {max(f.values(), key=lambda tr: tr.cx).track_id for f in out[1:]}
    assert len(ids_left) == 1 and len(ids_right) == 1 and ids_left != ids_right
    assert all(len(f) == 2 for f in out[1:])
    assert all(isinstance(tr, Track) and tr.height == 80.0 for f in out for tr in f.values())


def test_new_track_appears_on_its_second_frame():
    out = run(FaceTracker(CFG, FPS), [[], [], [box(100)], [box(100)], [box(100)]])
    assert [len(f) for f in out] == [0, 0, 0, 1, 1]


def test_crossing_paths_keep_ids():
    frames = [[box(100 + 4 * i), box(500 - 4 * i)] for i in range(120)]
    out = run(FaceTracker(CFG, FPS), frames)
    left_id, right_id = (tr.track_id for tr in sorted(out[5].values(), key=lambda tr: tr.cx))
    end_left_id, end_right_id = (tr.track_id for tr in sorted(out[-1].values(), key=lambda tr: tr.cx))
    assert (end_left_id, end_right_id) == (right_id, left_id)          # the ids crossed with the people
    assert all(len(f) == 2 for f in out[1:])


@pytest.mark.parametrize("gap_s", [1.0, 1.5, 2.0])
def test_track_survives_a_gap(gap_s):
    frames = [[box(100)]] * 10 + [[]] * int(gap_s * FPS) + [[box(100)]] * 3
    out = run(FaceTracker(CFG, FPS), frames)
    before = list(out[9])
    after = list(out[-1])
    assert before == after and len(before) == 1


def test_track_survives_a_gap_after_it_stops_moving():
    """Moving at 7 px/frame, then hidden 1.5 s, then back where it was last seen: the lost track's
    velocity must not carry its prediction a face-width away."""
    frames = [[box(100 + 7 * i)] for i in range(20)] + [[]] * 45 + [[box(100 + 7 * 19)]] * 3
    out = run(FaceTracker(CFG, FPS), frames)
    assert list(out[19]) == list(out[-1])
    # without the decay (and without the take-over, which would rescue it too) the same scenario breaks
    out = run(FaceTracker({**CFG, "lost_velocity_decay": 1.0, "reacquire_radius_h": 0.0}, FPS), frames)
    assert list(out[19]) != list(out[-1])
    # the decay alone is enough here
    out = run(FaceTracker({**CFG, "reacquire_radius_h": 0.0}, FPS), frames)
    assert list(out[19]) == list(out[-1])


def test_track_hidden_then_back_a_face_height_away_keeps_its_id():
    """Hidden 1.5 s and back 1.2 face heights to the side: no box overlap, so ByteTrack alone would
    start a new track; the take-over gives it the old id."""
    frames = [[box(100, 100)]] * 10 + [[]] * 45 + [[box(100, 100 + 96)]] * 3
    out = run(FaceTracker(CFG, FPS), frames)
    assert list(out[9]) == list(out[-1])
    out = run(FaceTracker({**CFG, "reacquire_radius_h": 0.0}, FPS), frames)
    assert list(out[9]) != list(out[-1])


def test_take_over_does_not_steal_from_a_person_still_on_screen():
    a, b = box(100), box(400)
    frames = [[a, b]] * 10 + [[b]] * 45 + [[b, box(100 + 90)]] * 3 + [[b, box(100 + 90), box(400 + 90, 300)]] * 3
    out = run(FaceTracker(CFG, FPS), frames)
    a_id, b_id = (tr.track_id for tr in sorted(out[9].values(), key=lambda tr: tr.cx))
    assert sorted(out[57].values(), key=lambda tr: tr.cx)[0].track_id == a_id     # A back, shifted: same id
    assert b_id in out[-1]                                                          # B never left
    newcomer = [tr for tr in out[-1].values() if tr.track_id not in (a_id, b_id)]
    assert len(newcomer) == 1 and len(out[-1]) == 3                                 # C next to B gets a new id


def test_track_expires_after_max_lost():
    frames = [[box(100)]] * 10 + [[]] * int(3.0 * FPS) + [[box(100)]] * 3
    out = run(FaceTracker(CFG, FPS), frames)
    assert list(out[9]) != list(out[-1])                                # a new id: that is re-ID's job


def test_gap_in_seconds_holds_at_other_frame_rates():
    fps = 60
    tr = FaceTracker(CFG, fps)
    frames = [[box(100)]] * 10 + [[]] * int(1.9 * fps) + [[box(100)]] * 3
    out = [{t.track_id for t in tr.update(d, FRAME, i / fps)} for i, d in enumerate(frames)]
    assert out[9] == out[-1]


def test_keypoints_and_scores_ride_along_and_low_scores_do_not_start_tracks():
    kps = np.array([[10.0, 20.0], [30.0, 20.0], [20.0, 35.0]])
    out = run(FaceTracker(CFG, FPS), [[box(100, kps=kps, score=0.7), box(400, score=0.3)]] * 5)
    assert all(len(f) == 1 for f in out[1:])
    tr = next(iter(out[-1].values()))
    assert tr.score == 0.7 and tr.keypoints is kps and tr.bbox == (100.0, 100.0, 160.0, 180.0)


def test_reset_forgets_tracks_but_keeps_counting():
    tr = FaceTracker(CFG, FPS)
    first = run(tr, [[box(100)]] * 5)[-1]
    tr.reset()
    second = run(tr, [[box(100)]] * 5)[-1]
    assert list(first) != list(second)
