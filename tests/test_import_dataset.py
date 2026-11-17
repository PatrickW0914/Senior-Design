"""Dataset converters, on small made-up files in each dataset's real format."""

import sys
import wave

import cv2
import numpy as np
import pytest

from config import REPO_ROOT
from labels import FaceLabels, Labels

sys.path.insert(0, str(REPO_ROOT / "tools"))
import import_dataset as imp  # noqa: E402

FPS = 25


def make_video(path, seconds=4.0, size=(64, 48), fps=FPS, fourcc="mp4v"):
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
    for i in range(int(seconds * fps)):
        frame = np.zeros((size[1], size[0], 3), np.uint8)
        frame[:, i % (size[0] - 4):i % (size[0] - 4) + 4] = 255       # a bar whose position is the frame number
        w.write(frame)
    w.release()
    return path


def bar_position(frame):
    """Left edge of the bright bar: survives lossy encoding, unlike a brightness value."""
    cols = frame.mean(axis=(0, 2))
    return int(np.argmax(cols > 0.5 * cols.max()))


def frames_in(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(3)), int(cap.get(4)))
    cap.release()
    return n, size


def test_merge_and_cut():
    words = [(1.0, 1.2), (1.3, 1.6), (2.0, 2.4), (3.5, 3.9), (3.95, 4.0)]
    assert imp.merge_stretches(words) == [(1.0, 2.4), (3.5, 4.0)]                # gaps under 0.5 s are bridged
    assert imp.merge_stretches(words, gap=0.2) == [(1.0, 1.6), (2.0, 2.4), (3.5, 4.0)]
    assert imp.cut([(1.0, 2.4), (3.5, 4.0)], 2.0, 3.75) == [(0.0, pytest.approx(0.4)), (1.5, 1.75)]
    assert imp.cut([(1.0, 2.4)], 5.0, None) == []


def test_trim_video_cuts_exact_frames(tmp_path):
    src = make_video(tmp_path / "src.mp4", seconds=4.0)
    fps, w, h, n, t0 = imp.trim_video(src, tmp_path / "out" / "cut.mp4", start=1.0, duration=2.0)
    assert (fps, w, h, n, t0) == (FPS, 64, 48, 50, 1.0)
    assert frames_in(tmp_path / "out" / "cut.mp4") == (50, (64, 48))
    cap = cv2.VideoCapture(str(tmp_path / "out" / "cut.mp4"))
    assert abs(bar_position(cap.read()[1]) - 25) <= 1                            # source frame 25 comes first
    cap.release()
    assert imp.trim_video(src, tmp_path / "all.mp4")[3] == 100


# --- AMI ----------------------------------------------------------------------------------
MEETINGS = """<?xml version="1.0" encoding="ISO-8859-1"?>
<nite:root nite:id="m" xmlns:nite="http://nite.sourceforge.net/">
  <meeting nite:id="meet_1" observation="XX1000a" duration="4">
    <speaker nite:id="XX1000a_1" channel="0" nxt_agent="A" camera="Closeup2" global_name="FEE001" role="PM"/>
    <speaker nite:id="XX1000a_2" channel="1" nxt_agent="B" camera="Closeup1" global_name="MEE002" role="ID"/>
  </meeting>
</nite:root>"""


def words_xml(agent, words):
    body = "".join(f'<w nite:id="w{i}" starttime="{a}" endtime="{b}">w</w>' for i, (a, b) in enumerate(words))
    return (f'<?xml version="1.0"?><nite:root nite:id="XX1000a.{agent}.words" xmlns:nite="http://nite.sourceforge.net/">'
            f'<vocalsound nite:id="v0" starttime="0.1" endtime="0.9" type="laugh"/>'
            f'<w nite:id="p0" starttime="1.5" endtime="1.5" punc="true">.</w>{body}<gap nite:id="g0"/></nite:root>')


@pytest.fixture
def ami(tmp_path):
    src = tmp_path / "ami"
    (src / "annotations" / "corpusResources").mkdir(parents=True)
    (src / "annotations" / "words").mkdir()
    (src / "annotations" / "corpusResources" / "meetings.xml").write_text(MEETINGS)
    (src / "annotations" / "words" / "XX1000a.A.words.xml").write_text(words_xml("A", [(1.0, 1.3), (1.4, 1.8), (3.0, 3.4)]))
    (src / "annotations" / "words" / "XX1000a.B.words.xml").write_text(words_xml("B", [(2.0, 2.6)]))
    for stream in ("Closeup1", "Closeup2", "Corner"):
        make_video(src / "XX1000a" / f"XX1000a.{stream}.avi", fourcc="MJPG")
    for name, value in [("Mix-Headset", 1000)] + [(f"Array1-0{i}", 100 * i) for i in range(1, 9)]:
        with wave.open(str(src / "XX1000a" / f"XX1000a.{name}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(np.full(4 * 16000, value, "<i2").tobytes())
    return src


def test_ami_parsing(ami):
    sp = imp.ami_speakers(ami / "annotations" / "corpusResources" / "meetings.xml", "XX1000a")
    assert sp["A"]["camera"] == "Closeup2" and sp["B"]["name"] == "MEE002" and set(sp) == {"A", "B"}
    with pytest.raises(KeyError):
        imp.ami_speakers(ami / "annotations" / "corpusResources" / "meetings.xml", "ZZ9999z")
    words = imp.ami_word_times(ami / "annotations" / "words" / "XX1000a.A.words.xml")
    assert words == [(1.0, 1.3), (1.4, 1.8), (3.0, 3.4)]                         # no laugh, punctuation or gap


def test_ami_import_closeup_wide_and_audio(ami, tmp_path, capsys):
    out = tmp_path / "data"
    assert imp.main(["ami", "--meeting", "XX1000a", "--src", str(ami), "--views", "closeup2", "corner", "closeup4",
                     "--audio", "--start", "1", "--duration", "2.5", "--data-dir", str(out)]) == 0
    assert frames_in(out / "videos" / "ami_xx1000a_closeup2.mp4")[0] == 62       # 2.5 s at 25 fps, rounded
    close = Labels.load(out / "labels" / "ami_xx1000a_closeup2.csv")             # speaker A only, as person 1
    assert [(s.t_start, s.t_end, s.person) for s in close.sorted()] == [(0.0, 0.8, 1), (2.0, 2.4, 1)]
    wide = Labels.load(out / "labels" / "ami_xx1000a_corner.csv")                # A and B as persons 1 and 2
    assert [(s.t_start, s.t_end, s.person) for s in wide.sorted()] == [(0.0, 0.8, 1), (1.0, 1.6, 2), (2.0, 2.4, 1)]
    with wave.open(str(out / "audio" / "ami_xx1000a_array1.wav")) as w:
        assert (w.getnchannels(), w.getframerate(), w.getnframes()) == (8, 16000, 40000)
        first = np.frombuffer(w.readframes(1), "<i2")
        assert first.tolist() == [100, 200, 300, 400, 500, 600, 700, 800]        # channels in array order
    with wave.open(str(out / "audio" / "ami_xx1000a_mix.wav")) as w:
        assert w.getnchannels() == 1 and w.getnframes() == 40000
    captured = capsys.readouterr()
    assert "closeup4" in captured.err and "missing" in captured.err              # no such camera file: skipped
    assert "not left-to-right" in captured.out


# --- AVA ----------------------------------------------------------------------------------
def test_ava_import(tmp_path):
    video = make_video(tmp_path / "ava" / "abc-XYZ_123.mkv", seconds=4.0, size=(200, 100))
    rows = []
    for i in range(75):                                                          # one face for 3 s, another for 2 s
        ts = 1.0 + i / FPS                                                       # the labels start 1 s into the film
        label = "SPEAKING_AUDIBLE" if i < 25 else "NOT_SPEAKING"
        rows.append(f"abc-XYZ_123,{ts:.2f},0.1,0.2,0.3,0.6,{label},abc-XYZ_123_0900_0960:1")
        if i < 50:
            rows.append(f"abc-XYZ_123,{ts:.2f},0.6,0.2,0.8,0.6,SPEAKING_NOT_AUDIBLE,abc-XYZ_123_0900_0960:2")
    labels = tmp_path / "ava" / "abc-XYZ_123-activespeaker.csv"
    labels.write_text("\n".join(rows) + "\n")
    parsed = imp.ava_rows(labels)
    assert len(parsed) == 125 and parsed[0][5] == "SPEAKING_AUDIBLE" and parsed[0][0] == 1.0
    out = tmp_path / "data"
    assert imp.main(["ava", "--id", "abc-XYZ_123", "--video", str(video), "--labels", str(labels),
                     "--duration", "2", "--data-dir", str(out)]) == 0
    name = "ava_abc_xyz_123_1s"                                                  # cut from where the labels start
    assert frames_in(out / "videos" / f"{name}.mp4") == (50, (200, 100))
    faces = FaceLabels.load(out / "labels" / f"{name}.faces.csv")
    assert faces.people() == {1, 2} and faces.span()[0] == 0.0 and faces.span()[1] < 2.0
    first = sorted(faces.at(0.0), key=lambda r: r.person)
    assert (first[0].x1, first[0].y1, first[0].x2, first[0].y2) == (20.0, 20.0, 60.0, 60.0)   # scaled to 200x100
    assert first[0].speaking and first[1].speaking and first[0].box == "face"    # audible or not, lips move
    assert faces.speaking_for(1.5, 40, 40) is False                              # after the first second
    assert imp.main(["ava", "--id", "nope", "--src", str(tmp_path / "ava"), "--data-dir", str(out)]) == 1


# --- RealVAD ------------------------------------------------------------------------------
def test_realvad_import(tmp_path):
    folder = tmp_path / "realvad"
    folder.mkdir()
    # panelist 1: frames 10..59; the last two numbers are the second corner, and one run is repeated out of order
    box = "\n".join(f"{f} 100 200 300 400" for f in list(range(10, 60)) + list(range(20, 25)))
    (folder / "Panelist1_bbox.txt").write_text(box + "\n")
    (folder / "Panelist1_VAD.txt").write_text("\n".join(f"{f}\t{int(f >= 35)}" for f in range(10, 60)) + "\n")
    (folder / "Panelist2_bbox.txt").write_text("\n".join(f"{f} 700 200 900 420" for f in range(10, 60)) + "\n")
    (folder / "Panelist2_VAD.txt").write_text("\n".join(f"{f}\t0" for f in range(10, 40)) + "\n")
    boxes, vad = imp.realvad_panelist(folder, 1)
    assert boxes[20] == (100.0, 200.0, 300.0, 400.0) and vad[34] is False and vad[35] is True and len(boxes) == 50
    video = make_video(tmp_path / "panel.mp4", seconds=4.0, size=(640, 360))     # half the labeled size
    assert imp.main(["realvad", "--video", str(video), "--src", str(folder), "--data-dir", str(tmp_path / "data")]) == 0
    faces = FaceLabels.load(tmp_path / "data" / "labels" / "realvad_panel.faces.csv")
    assert faces.people() == {1, 2} and len(faces.rows) == 50 + 30
    r = next(x for x in faces.at(35 / FPS) if x.person == 1)
    assert (r.x1, r.y1, r.x2, r.y2) == (50.0, 100.0, 150.0, 200.0) and r.speaking and r.box == "body"   # halved
    assert faces.speaking_for(20 / FPS, 100, 120) is False and faces.speaking_for(50 / FPS, 100, 120) is True
    assert (tmp_path / "data" / "videos" / "realvad_panel_check.jpg").is_file()
    assert imp.main(["realvad", "--video", str(tmp_path / "none.mp4"), "--src", str(folder)]) == 1
