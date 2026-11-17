#!/usr/bin/env python3
"""main.py - video-only pipeline: capture -> detect -> track -> azimuth + speaking score.

Every frame becomes a list of FaceObs (the record av_event_log expects), written with its
timestamp to <output.dir>/<prefix>_faces.jsonl. When the mic array arrives, the same
(t, faces) pair goes to av_event_log.AVSpeakerPipeline.on_video_frame(t, faces).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv_env  # noqa: F401  (must stay above cv2)
import cv2

from av_event_log import FaceObs, SessionClock, _json_default
from capture import VideoSource
from config import DEFAULT_CONFIG, load_config, repo_path
from detect import FaceDetector
from geometry import AzimuthModel
from overlay import Overlay, VideoSaver
from speaking import SpeakingScorer, SpeakingState
from track import FaceTracker


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Video-only speaker detection pipeline.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to config.yaml (default: %(default)s)")
    p.add_argument("--source", help="camera index (e.g. 0) or video file path; overrides 'source' in the config")
    p.add_argument("--show", action="store_true", help="show the debug overlay window (press q to quit)")
    p.add_argument("--save-video", nargs="?", const="", metavar="MP4",
                   help="save the annotated video; without a path: <output.dir>/<prefix>_overlay.mp4")
    p.add_argument("--max-frames", type=int, help="stop after this many frames")
    p.add_argument("--out-prefix", help="output file prefix; overrides output.prefix in the config")
    return p.parse_args(argv)


def parse_source(src: int | str) -> int | str:
    """'0' -> camera 0; a file path is tried as given, then relative to the repo root."""
    if isinstance(src, int) or str(src).isdigit():
        return int(src)
    return src if Path(src).exists() else str(repo_path(src))


def run(source: int | str, args: argparse.Namespace, cfg: dict) -> Path:
    clock = SessionClock(camera_latency_s=cfg["camera"]["latency_s"])
    detector = FaceDetector(cfg["detect"])
    scorer = SpeakingScorer(cfg["speaking"])
    state = SpeakingState(cfg["speaking"])
    scoring = None              # None until the scorer has been tried, then True, or False if not implemented yet

    out_dir = repo_path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or cfg["output"]["prefix"]
    out_path = out_dir / f"{prefix}_faces.jsonl"
    saver = None

    with VideoSource(source, cfg["camera"], clock) as video, open(out_path, "w", encoding="utf-8") as fh:
        fps = video.fps or cfg["camera"]["fps"]
        tracker = FaceTracker(cfg["track"], fps=fps)
        angles = AzimuthModel.from_config(cfg["camera"], video.width, video.height)
        view = Overlay(cfg.get("overlay"), angles) if args.show or args.save_video is not None else None
        if args.save_video is not None:
            saver = VideoSaver(args.save_video or out_dir / f"{prefix}_overlay.mp4", fps)
        try:
            for i, (t, frame) in enumerate(video):
                if args.max_frames is not None and i >= args.max_frames:
                    break
                tracks = tracker.update(detector.detect(frame), frame, t)
                probs = {}
                for trk in tracks:
                    if scoring is not False:
                        try:
                            probs[trk.track_id] = scorer.score(trk, frame, t)
                            scoring = True
                        except NotImplementedError as e:
                            print(f"note: {e} is not implemented yet; running without speaking scores", file=sys.stderr)
                            scoring = False
                faces = [FaceObs(trk.track_id, angles.azimuth(trk.cx, trk.cy), probs.get(trk.track_id, 0.0), trk.bbox)
                         for trk in tracks]
                fh.write(json.dumps({"t": round(t, 3), "faces": [asdict(f) for f in faces]},
                                    default=_json_default) + "\n")
                if view is not None:
                    img = view.draw(frame, t, faces, mouth=scorer.mouth if scoring else None,
                                    speaking=state.update(probs) if scoring else None,     # None = not known
                                    note="video only, no speaking stage yet" if scoring is False else "")
                    if saver is not None:
                        saver.write(img)
                    if args.show:
                        cv2.imshow("speaker-agent", img)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
        finally:
            if saver is not None:
                saver.close()
    if args.show:
        cv2.destroyAllWindows()
    if saver is not None:
        print(f"Wrote {saver.path} ({saver.frames} frames)")
    return out_path


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    source = parse_source(args.source if args.source is not None else cfg["source"])
    try:
        out_path = run(source, args, cfg)
    except NotImplementedError as e:
        print(f"Pipeline stage not implemented yet -> {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:          # a model file or the clip is missing: say which, no traceback
        print(e, file=sys.stderr)
        return 2
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
