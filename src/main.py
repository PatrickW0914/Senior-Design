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

import cv2

import overlay
from av_event_log import FaceObs, SessionClock, _json_default
from capture import open_source
from config import DEFAULT_CONFIG, load_config, repo_path
from detect import FaceDetector
from geometry import pixel_to_azimuth
from speaking import SpeakingScorer
from track import FaceTracker


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Video-only speaker detection pipeline.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to config.yaml (default: %(default)s)")
    p.add_argument("--source", help="camera index (e.g. 0) or video file path; overrides 'source' in the config")
    p.add_argument("--show", action="store_true", help="show the debug overlay window (press q to quit)")
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
    tracker = FaceTracker(cfg["track"])
    scorer = SpeakingScorer(cfg["speaking"])
    hfov = cfg["camera"]["hfov_deg"]

    out_dir = repo_path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_prefix or cfg['output']['prefix']}_faces.jsonl"

    with open(out_path, "w", encoding="utf-8") as fh:
        for i, (t, frame) in enumerate(open_source(source, cfg["camera"], clock)):
            if args.max_frames is not None and i >= args.max_frames:
                break
            tracks = tracker.update(detector.detect(frame), frame, t)
            faces = [FaceObs(trk.track_id, pixel_to_azimuth(trk.cx, frame.shape[1], hfov),
                             scorer.score(trk, frame, t), trk.bbox) for trk in tracks]
            fh.write(json.dumps({"t": round(t, 3), "faces": [asdict(f) for f in faces]},
                                default=_json_default) + "\n")
            if args.show:
                cv2.imshow("speaker-agent", overlay.draw(frame, t, faces))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    if args.show:
        cv2.destroyAllWindows()
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
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
