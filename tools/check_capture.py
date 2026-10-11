#!/usr/bin/env python3
"""check_capture.py - Task 2 check: timestamps monotonic? real FPS? dropped frames?

  python tools/check_capture.py --source data/videos/clip.mp4
  python tools/check_capture.py --source 0 --seconds 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from av_event_log import SessionClock  # noqa: E402
from capture import VideoSource, frame_timing  # noqa: E402
from config import load_config  # noqa: E402
from main import parse_source  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", required=True, help="camera index or video file")
    p.add_argument("--seconds", type=float, default=60.0, help="camera only: how long to record (default 60)")
    args = p.parse_args(argv)

    cfg = load_config()
    clock = SessionClock(camera_latency_s=cfg["camera"]["latency_s"])
    ts = []
    with VideoSource(parse_source(args.source), cfg["camera"], clock) as video:
        print(f"source={video.source!r} camera={video.is_camera} {video.width}x{video.height} "
              f"reported fps={video.fps:.2f}")
        for t, _ in video:
            ts.append(t)
            if video.is_camera and t - ts[0] >= args.seconds:
                break

    s = frame_timing(ts, nominal_fps=video.fps or None)
    print(f"frames={s.n_frames}  span={s.span_s:.3f}s  measured fps={s.measured_fps:.2f}  "
          f"median interval={s.median_interval_s * 1000:.1f}ms  max gap={s.max_gap_s * 1000:.1f}ms")
    print(f"monotonic={s.monotonic}  long gaps (> 1.5x frame interval)={s.long_gaps}  "
          f"missing frames={s.missing}")
    ok = s.monotonic and s.n_frames > 1
    if not video.is_camera:
        end = ts[-1] + 1.0 / video.fps if ts else 0.0
        print(f"file: decoded {s.n_frames}/{video.frame_count} frames, "
              f"last t + 1 frame = {end:.3f}s, header duration = {video.duration_s:.3f}s")
        ok = ok and s.n_frames == video.frame_count
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
