"""Task 7: the overlay draws what it should, where it should, and the video saver writes a playable file."""

import cv2
import numpy as np
import pytest
import yaml

import main
from av_event_log import FaceObs
from config import load_config
from detect import Detection
from geometry import AzimuthModel
from overlay import Overlay, VideoSaver, draw
from speaking import SpeakingState

W, H, STRIP = 1280, 720, 90
CFG = {"width": 1280, "graph_seconds": 3.0, "strip_height": STRIP, "names": {}}
A = FaceObs(1, -12.3, 0.8, (300, 200, 460, 400))
B = FaceObs(2, 15.0, 0.2, (800, 220, 960, 420))


def frame(w=W, h=H):
    return np.full((h, w, 3), 80, np.uint8)


def is_green(px):
    b, g, r = (int(v) for v in px)
    return g > 150 and g > r + 60 and g > b + 60


def is_grey(px):
    b, g, r = (int(v) for v in px)
    return min(b, g, r) > 150 and max(b, g, r) - min(b, g, r) < 20


def count(img, pred):
    b, g, r = (img[..., i].astype(int) for i in range(3))
    return int(pred(b, g, r).sum())


def red(b, g, r):
    return (r > 180) & (g < 90) & (b < 90)


def yellow(b, g, r):
    return (r > 200) & (g > 170) & (b < 90)


def test_size_is_frame_plus_strip_scaled_and_never_enlarged():
    ov = Overlay(CFG)
    assert ov.draw(frame(), 0.0, []).shape == (H + STRIP, W, 3)
    assert ov.draw(frame(3840, 2160), 0.0, []).shape == (H + STRIP, W, 3)       # 4K comes down to 1280
    assert ov.draw(frame(640, 360), 0.0, []).shape == (360 + STRIP, 640, 3)     # small stays small
    assert ov.output_size(3840, 2160) == (W, H + STRIP)


def test_box_is_green_when_speaking_grey_otherwise():
    ov = Overlay(CFG)
    img = ov.draw(frame(), 0.0, [A, B], speaking={1})
    assert is_green(img[300, 300]) and is_grey(img[320, 800])                   # left edges of the two boxes
    img = ov.draw(frame(), 0.1, [A, B], speaking=None)                          # speaking not known: nobody green
    assert is_grey(img[300, 300]) and is_grey(img[320, 800])
    assert tuple(img[600, 1200]) == (80, 80, 80)                                # the picture itself is untouched


def test_boxes_follow_the_scale():
    big = FaceObs(1, 0.0, 0.9, (900, 600, 1380, 1200))                          # in 4K pixels
    img = Overlay(CFG).draw(frame(3840, 2160), 0.0, [big], speaking={1})
    assert is_green(img[300, 300])                                              # 900/3 = 300


def test_overlap_banner_only_when_two_speak():
    ov = Overlay(CFG)
    banner = (slice(H - 60, H - 8), slice(W // 2 - 130, W // 2 + 130))         # bottom centre, above the strip
    both = ov.draw(frame(), 0.0, [A, B], speaking={1, 2})
    assert count(both[banner], red) > 2000
    assert count(both[:H, :6], red) > 500                                       # ... and the red frame, left edge
    one = ov.draw(frame(), 0.1, [A, B], speaking={1})
    assert count(one[banner], red) == 0 and count(one[:H], red) == 0
    assert count(ov.draw(frame(), 0.2, [A], speaking={1, 2})[:H], red) == 0     # #2 is not in the frame


def test_mouth_graph_draws_a_line_and_says_na_without_data():
    ov = Overlay(CFG)
    below_a = (slice(400, 490), slice(290, 480))
    img = ov.draw(frame(), 0.0, [A], mouth=None, speaking=None)
    assert count(img[below_a], yellow) == 0
    for i in range(30):
        img = ov.draw(frame(), i / 30, [A], mouth={1: 0.02 + 0.02 * (i % 2)}, speaking={1})
    assert count(img[below_a], yellow) > 30
    # without a mouth value but with speaking known, the probability is graphed instead
    ov2 = Overlay(CFG)
    for i in range(30):
        img = ov2.draw(frame(), i / 30, [A], speaking=set())
    assert count(img[below_a], yellow) > 30


def test_graph_history_ages_out_and_resets_when_time_goes_back():
    ov = Overlay(CFG)
    for i in range(10):
        ov.draw(frame(), i / 10, [A], mouth={1: 0.5})
    assert len(ov._hist[1]) == 10
    ov.draw(frame(), 10.0, [B], mouth={2: 0.1})            # 9 s later, #1 gone
    assert 1 not in ov._hist and len(ov._hist[2]) == 1
    ov.draw(frame(), 0.0, [A], mouth={1: 0.5})             # a new clip starts at 0
    assert list(ov._hist) == [1] and len(ov._hist[1]) == 1


def test_strip_has_axis_markers_and_takes_a_spectrum():
    angles = AzimuthModel.from_hfov(W, H, 70.4)
    ov = Overlay(CFG, angles)
    img = ov.draw(frame(), 0.0, [A], speaking={1})
    strip = img[H:]
    assert strip.shape[0] == STRIP and strip.mean() > 26                         # not just the dark fill
    assert count(strip[:20, 360:400], lambda b, g, r: (g > 150) & (g > r + 60)) > 20   # green marker under face #1
    cyan = lambda b, g, r: (b > 200) & (g > 150) & (r < 80)                      # noqa: E731
    assert count(strip, cyan) == 0
    grid = np.arange(-90, 91, 2)
    img = ov.draw(frame(), 0.1, [A], speaking={1}, doa=(grid, np.exp(-0.5 * ((grid + 12) / 6) ** 2)))
    assert count(img[H:], cyan) > 100


def test_names_change_the_label_and_one_call_draw_works():
    ov = Overlay({**CFG, "names": {1: "Alice"}})
    named = ov.draw(frame(), 0.0, [A], speaking={1})
    plain = Overlay(CFG).draw(frame(), 0.0, [A], speaking={1})
    assert (named[150:200] != plain[150:200]).any()
    assert is_green(draw(frame(), 0.0, [A, B])[300, 300])                        # asd_prob 0.8 >= 0.5


def test_speaking_state_hysteresis():
    st = SpeakingState({"on_threshold": 0.6, "off_threshold": 0.4})
    assert st.update({1: 0.5}) == set()            # between the thresholds: stays off
    assert st.update({1: 0.7}) == {1}
    assert st.update({1: 0.5}) == {1}              # ... and stays on
    assert st.update({1: 0.3}) == set()
    assert st.update({1: 0.9, 2: 0.9}) == {1, 2}
    assert st.update({2: 0.9}) == {2}              # #1 left the frame


def test_video_saver_writes_a_playable_mp4(tmp_path):
    path = tmp_path / "sub" / "out.mp4"
    ov = Overlay(CFG)
    with VideoSaver(path, 25) as saver:
        for i in range(8):
            saver.write(ov.draw(frame(), i / 25, [A], speaking={1}))
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    assert n == 8 == saver.frames
    assert (int(cap.get(3)), int(cap.get(4))) == (W, H + STRIP) and cap.get(5) == pytest.approx(25)
    cap.release()


def test_main_saves_an_overlay_video_without_a_speaking_stage(tmp_path, monkeypatch, capsys):
    """End to end with a stand-in detector: main runs through although Task 8 is not there."""
    clip = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 20, (320, 240))
    for i in range(12):
        w.write(np.full((240, 320, 3), 60 + i, np.uint8))
    w.release()
    cfg = load_config()
    cfg["output"]["dir"] = str(tmp_path / "out")
    cfg["camera"]["calibration"] = ""
    cfg_path = tmp_path / "config.yaml"
    yaml.safe_dump(cfg, open(cfg_path, "w"))

    class FakeDetector:
        def __init__(self, cfg):
            pass

        def detect(self, frame):
            return [Detection((100.0, 60.0, 180.0, 160.0), 0.9)]

    monkeypatch.setattr(main, "FaceDetector", FakeDetector)
    assert main.main(["--config", str(cfg_path), "--source", str(clip), "--save-video", "--out-prefix", "t"]) == 0
    lines = (tmp_path / "out" / "t_faces.jsonl").read_text().splitlines()
    assert len(lines) == 12 and '"track_id": 1' in lines[-1]
    cap = cv2.VideoCapture(str(tmp_path / "out" / "t_overlay.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    assert (int(cap.get(3)), int(cap.get(4))) == (320, 240 + cfg["overlay"]["strip_height"])
    cap.release()
    assert "running without speaking scores" in capsys.readouterr().err
