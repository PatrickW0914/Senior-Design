"""Task 9: video-only fusion, the segment builder's corrections, the config wiring and the segment check."""

import json
import sys
import warnings

import cv2
import numpy as np
import pytest
import yaml

import main
from av_event_log import VIDEO_ONLY_WEIGHTS, AVSpeakerPipeline, FaceObs, FrameResult, SegmentBuilder
from config import REPO_ROOT, load_config
from detect import Detection
from labels import Labels, Segment

sys.path.insert(0, str(REPO_ROOT / "eval"))
import segments_vs_labels as sv  # noqa: E402

GRID = np.arange(-90, 91, 2.0)
FPS = 25


def feed(pipe, spans, seconds=10.0):
    """spans: {track_id: (start, end)} of scripted speaking; everyone is visible the whole time."""
    for i in range(int(seconds * FPS)):
        t = i / FPS
        pipe.on_video_frame(t, [FaceObs(k, -20.0 if k == 1 else 25.0, 0.95 if a <= t < b else 0.05)
                                for k, (a, b) in spans.items()])
    return pipe.close()


def test_default_weights_without_audio_mark_nobody_and_say_why(tmp_path):
    """The failure Task 9 is about: capped at w_asd = 0.5, below the 0.6 needed to start."""
    pipe = AVSpeakerPipeline(GRID, out_prefix=str(tmp_path / "av"))
    with pytest.warns(RuntimeWarning, match="no audio frames"):
        segments, _ = feed(pipe, {1: (1.0, 5.0)})
    assert segments == [] and not pipe.fusion.video_only


def test_video_only_segments_overlap_and_a_clean_log(tmp_path):
    pipe = AVSpeakerPipeline(GRID, out_prefix=str(tmp_path / "clip"), names={1: "Alice"}, video_only=True)
    assert pipe.fusion.video_only
    with warnings.catch_warnings():
        warnings.simplefilter("error")                              # video-only must not warn about audio
        segments, unassigned = feed(pipe, {1: (1.0, 5.0), 2: (3.0, 7.0)})
    assert unassigned == [] and [s.track_id for s in segments] == [1, 2]
    a, b = segments
    assert abs(a.t_start - 1.0) < 0.35 and abs(a.t_end - 5.0) < 0.35     # the smoothing costs about 0.25 s
    assert abs(b.t_start - 3.0) < 0.35 and abs(b.t_end - 7.0) < 0.35
    assert a.overlap and b.overlap and a.source == "av" and a.text == "" and b.text == ""
    assert (a.speaker, b.speaker) == ("Alice", "Speaker A") and a.azimuth_deg == pytest.approx(-20.0)

    payload = json.loads((tmp_path / "clip_transcript.json").read_text(encoding="utf-8"))
    assert set(payload) == {"speakers", "segments", "unassigned_words"} and payload["unassigned_words"] == []
    assert [s["speaker"] for s in payload["segments"]] == ["Alice", "Speaker A"]
    assert all(s["text"] == "" and s["voice_id"] is None for s in payload["segments"])
    events = [json.loads(line) for line in (tmp_path / "clip_events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["event"] for e in events] == ["session_start", "segment", "segment", "session_end"]
    assert events[0]["mode"] == "video_only" and events[-1]["n_segments"] == 2
    assert "Alice (-20 deg) [overlap]:" in (tmp_path / "clip_transcript.txt").read_text(encoding="utf-8")


def test_the_form_from_the_task_and_the_flag_agree(tmp_path):
    by_kwargs = AVSpeakerPipeline(GRID, out_prefix=str(tmp_path / "a"),
                                  fusion_kwargs=dict(w_asd=1.0, w_doa=0.0, w_vad=0.0))
    by_flag = AVSpeakerPipeline(GRID, out_prefix=str(tmp_path / "b"), video_only=True,
                                fusion_kwargs=dict(w_asd=0.5, w_doa=0.4, tau_s=0.25))      # the flag wins
    assert by_kwargs.fusion.video_only and by_flag.fusion.video_only
    assert (by_flag.fusion.w_asd, by_flag.fusion.w_doa, by_flag.fusion.w_vad) == tuple(VIDEO_ONLY_WEIGHTS.values())
    spans = {1: (1.0, 4.0)}
    assert [(s.t_start, s.t_end) for s in feed(by_kwargs, spans)[0]] == [(s.t_start, s.t_end) for s in feed(by_flag, spans)[0]]


def frames(on_spans, seconds=6.0):
    for i in range(int(seconds * FPS)):
        t = i / FPS
        on = any(a <= t < b for a, b in on_spans)
        yield FrameResult(t, {1: 0.9 if on else 0.1}, {1: on}, {1: 10.0}, 0.0, False, False)


def built(builder, on_spans):
    segs = [s for fr in frames(on_spans) for s in builder.update(fr)] + builder.flush()
    return [(round(s.t_start, 2), round(s.t_end, 2)) for s in segs]


def test_segment_builder_corrections_and_gap():
    assert built(SegmentBuilder(), [(1.0, 3.0)]) == [(1.0, 2.96)]
    assert built(SegmentBuilder(onset_latency_s=0.2, release_latency_s=0.5), [(1.0, 3.0)]) == [(0.8, 2.46)]
    assert built(SegmentBuilder(release_latency_s=0.5), [(1.0, 1.6)]) == []          # a blip that was mostly tail
    assert built(SegmentBuilder(), [(1.0, 1.6)]) == [(1.0, 1.56)]
    assert built(SegmentBuilder(gap_merge_s=0.5), [(1.0, 2.0), (2.4, 3.0)]) == [(1.0, 2.96)]      # a 0.4 s dropout
    assert built(SegmentBuilder(gap_merge_s=0.3), [(1.0, 2.0), (2.4, 3.0)]) == [(1.0, 1.96), (2.4, 2.96)]
    assert built(SegmentBuilder(onset_latency_s=5.0), [(1.0, 3.0)]) == [(0.0, 2.96)]              # never before 0


def test_build_pipeline_follows_the_config(tmp_path):
    cfg = load_config()
    pipe = main.build_pipeline(cfg, str(tmp_path / "a"))
    assert pipe.fusion.video_only and cfg["fusion"]["mode"] == "video_only"
    assert pipe.builder.release_latency_s == cfg["fusion"]["asd_release_s"]
    assert pipe.builder.gap_merge_s == cfg["fusion"]["gap_merge_s"] and pipe.fusion.tau_s == cfg["fusion"]["tau_s"]
    av = main.build_pipeline(cfg, str(tmp_path / "b"), {"mode": "audio_visual", "gap_merge_s": 1.5})
    assert not av.fusion.video_only and av.fusion.w_asd == cfg["fusion"]["w_asd"] and av.builder.gap_merge_s == 1.5
    named = main.build_pipeline({**cfg, "overlay": {"names": {2: "Bob"}}}, str(tmp_path / "c"))
    assert named.log.label(2) == "Bob"
    with pytest.raises(ValueError):
        main.build_pipeline(cfg, str(tmp_path / "d"), {"mode": "audio"})
    for p in (pipe, av, named):
        p.close()


def test_compare_segments_with_labels():
    labels = Labels([Segment(1.0, 5.0, 1), Segment(8.0, 12.0, 1), Segment(14.0, 14.3, 1), Segment(20.0, 23.0, 1)])
    times = [i / FPS for i in range(30 * FPS)]
    cxs, seen = {7: [300.0] * len(times)}, {7: times}
    frontal = {7: [t for t in times if t < 18.0]}                       # turned away from 18 s on
    segs = [(7, 1.2, 3.0), (7, 3.8, 5.3), (7, 8.1, 12.9), (7, 16.0, 17.0)]
    r = sv.compare(segs, cxs, 1280, seen, frontal, labels, tol=0.5)
    assert (r["turns"], r["short"], r["found"], r["pieces"]) == (3, 1, 2, 3)           # the 20-23 s turn is missed
    assert (r["start_ok"], r["end_ok"], r["extra"]) == (2, 1, 1)                       # 12.9 is 0.9 s late
    assert r["start_err"] == pytest.approx([0.2, 0.1]) and r["end_err"] == pytest.approx([0.3, 0.9])
    assert r["lab_time"] == pytest.approx(11.3) and r["lab_frontal"] == pytest.approx(8.3, abs=0.1)
    assert r["covered"] == pytest.approx(1.8 + 1.2 + 3.9) and r["pred_right"] == pytest.approx(6.9)
    quiet = sv.compare([(7, 3.0, 4.0), (7, 9.0, 9.5)], cxs, 1280, seen, frontal, Labels(), tol=0.5)
    assert (quiet["turns"], quiet["extra"], quiet["pred_time"]) == (0, 2, 1.5)         # "nobody speaks": all extra
    assert sv.spans([0.0, 0.1, 0.2, 2.0, 2.1]) == [(0.0, 0.2), (2.0, 2.1)]


class FakeDetector:
    def __init__(self, cfg):
        pass

    def detect(self, frame):
        return [Detection((100.0, 60.0, 180.0, 160.0), 0.9)]


class FakeScorer:
    """Speaks from 0.5 s to 2.5 s."""

    def __init__(self, cfg):
        self.mouth = {}

    def score(self, track, frame, t):
        return 0.95 if 0.5 <= t < 2.5 else 0.05


def test_main_writes_transcript_and_event_log(tmp_path, monkeypatch, capsys):
    clip = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 20, (320, 240))
    for i in range(80):                                                  # 4 s
        w.write(np.full((240, 320, 3), 60 + i, np.uint8))
    w.release()
    cfg = load_config()
    cfg["output"]["dir"] = str(tmp_path / "out")
    cfg["camera"]["calibration"] = ""
    yaml.safe_dump(cfg, open(tmp_path / "config.yaml", "w"))
    monkeypatch.setattr(main, "FaceDetector", FakeDetector)
    monkeypatch.setattr(main, "SpeakingScorer", FakeScorer)
    assert main.main(["--config", str(tmp_path / "config.yaml"), "--source", str(clip), "--out-prefix", "t"]) == 0
    payload = json.loads((tmp_path / "out" / "t_transcript.json").read_text(encoding="utf-8"))
    assert len(payload["segments"]) == 1
    seg = payload["segments"][0]
    assert abs(seg["t_start"] - 0.5) < 0.5 and abs(seg["t_end"] - 2.5) < 0.5 and seg["text"] == "" and seg["track_id"] == 1
    events = [json.loads(line)["event"] for line in (tmp_path / "out" / "t_events.jsonl").read_text().splitlines()]
    assert events == ["session_start", "segment", "session_end"]
    assert (tmp_path / "out" / "t_transcript.txt").read_text(encoding="utf-8").startswith("[00:00:00.")
    assert "1 speaking segment(s)" in capsys.readouterr().out
