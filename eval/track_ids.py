#!/usr/bin/env python3
"""track_ids.py - Task 5 check: do track IDs stay put, and do they survive an occlusion?

  python eval/track_ids.py                                    # every clip in data/videos
  python eval/track_ids.py --clips two_person_overlap one_person_walking --video
  python eval/track_ids.py --out eval/results/track_ids.md

Runs detector + tracker on every frame. Per clip:
  people           from the clip name (one_/two_/three_person_...), the number of IDs to hope for
  ids              track IDs created; ids >2s: those seen for more than two seconds
  jumps            switch candidates: an ID whose centre moves more than 2 face heights (and 10% of
                   the width) between sightings at most 5 frames apart. On the podcast clips camera
                   cuts cause these; elsewhere they mean a swap or a wrong re-association.
  gap bridged      the longest run of frames an ID went unseen and then came back as the same ID
  occlusion test   the clip is run again three times with every detection suppressed for 1.5 s,
                   starting 25%, 50% and 75% in, as if someone walked in front. For each: were
                   the IDs seen just before back within 10 frames after, and how far (in face
                   heights) the faces moved meanwhile. A face that moved less than one face
                   height should keep its ID; one that walked off is re-ID's job.
--video writes eval/results/<clip>_tracks.mp4 (960 px wide) with boxes and IDs to eyeball.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv_env  # noqa: F401, E402
import cv2  # noqa: E402

from av_event_log import SessionClock  # noqa: E402
from capture import VideoSource  # noqa: E402
from config import load_config, repo_path  # noqa: E402
from detect import FaceDetector  # noqa: E402
from track import FaceTracker  # noqa: E402

OCCLUDE_S, OCCLUDE_AT, LONG_S = 1.5, (0.25, 0.50, 0.75), 2.0
PEOPLE = {"one": 1, "two": 2, "three": 3, "four": 4}


def color(track_id: int):
    rng = np.random.default_rng(track_id * 7919)
    return tuple(int(v) for v in rng.integers(80, 255, 3))


def detect_clip(path: Path, cfg: dict, writer=None, scale=1.0, size=None):
    """Detect once per frame (the slow part). Returns (detections per frame, times, fps, (w, h)),
    and with a writer also tracks the clean run and writes the annotated video."""
    detector = FaceDetector(cfg["detect"])
    dets_all, times = [], []
    with VideoSource(str(path), cfg["camera"], SessionClock()) as video:
        tracker = FaceTracker(cfg["track"], fps=video.fps) if writer is not None else None
        for i, (t, frame) in enumerate(video):
            dets = detector.detect(frame)
            dets_all.append(dets)
            times.append(t)
            if writer is not None:
                tracks = tracker.update(dets, frame, t)
                img = cv2.resize(frame, size) if scale < 1 else frame.copy()   # exactly the writer's size
                for tr in tracks:
                    x1, y1, x2, y2 = (int(v * scale) for v in tr.bbox)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color(tr.track_id), 2)
                    cv2.putText(img, f"#{tr.track_id}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                color(tr.track_id), 2, cv2.LINE_AA)
                cv2.putText(img, f"{t:6.2f}s  {len(tracks)} tracks", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2, cv2.LINE_AA)
                writer.write(img)
        return dets_all, times, video.fps, (video.width, video.height)


def track_clip(dets_all, times, fps, cfg, occlude: tuple[int, int] | None = None):
    """Run the tracker over cached detections. Returns per-frame (frame_index, [(id, cx, cy, h)])."""
    tracker = FaceTracker(cfg["track"], fps=fps)
    frames = []
    for i, (dets, t) in enumerate(zip(dets_all, times)):
        if occlude and occlude[0] <= i < occlude[1]:
            dets = []
        tracks = tracker.update(dets, None, t)
        frames.append((i, [(tr.track_id, tr.cx, 0.5 * (tr.bbox[1] + tr.bbox[3]), tr.height) for tr in tracks]))
    return frames


def analyse(frames, fps: float, width: int) -> dict:
    seen = defaultdict(list)                     # id -> [(frame, cx, cy, h)]
    for i, tracks in frames:
        for tid, cx, cy, h in tracks:
            seen[tid].append((i, cx, cy, h))
    jumps, gaps = [], {}
    for tid, obs in seen.items():
        best = 0
        for (i0, x0, y0, h0), (i1, x1, y1, h1) in zip(obs, obs[1:]):
            d = float(np.hypot(x1 - x0, y1 - y0))
            if i1 - i0 <= 5 and d > max(2 * max(h0, h1), 0.1 * width):
                jumps.append((tid, i0 / fps, d))
            best = max(best, i1 - i0 - 1)
        gaps[tid] = best
    long_ids = [tid for tid, obs in seen.items() if (obs[-1][0] - obs[0][0]) / fps > LONG_S]
    return dict(ids=len(seen), long_ids=len(long_ids), jumps=jumps,
                gap_frames=max(gaps.values()) if gaps else 0, gaps=gaps,
                frames_with_tracks=sum(1 for _, tr in frames if tr), n_frames=len(frames))


def occlusion_result(frames, clean_frames, occlude: tuple[int, int]) -> tuple[int, int, float]:
    """(ids seen just before the blackout, how many of them are back just after, how far the faces
    moved during the blackout in face heights, from the clean run)."""
    before = {tid for i, tr in frames if occlude[0] - 5 <= i < occlude[0] for tid, *_ in tr}
    after = {tid for i, tr in frames if occlude[1] <= i < occlude[1] + 10 for tid, *_ in tr}
    pos = {}
    for i, tr in clean_frames:
        if i == occlude[0] - 1 or i == occlude[1]:
            for tid, cx, cy, h in tr:
                pos.setdefault(tid, {})[i] = (cx, cy, h)
    moved = [np.hypot(p[occlude[1]][0] - p[occlude[0] - 1][0], p[occlude[1]][1] - p[occlude[0] - 1][1]) / p[occlude[0] - 1][2]
             for p in pos.values() if occlude[0] - 1 in p and occlude[1] in p]
    return len(before), len(before & after), float(max(moved)) if moved else float("nan")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--clips", nargs="*", help="clip names in data/videos (default: all)")
    p.add_argument("--video", action="store_true", help="write annotated videos to eval/results/")
    p.add_argument("--no-occlusion", action="store_true", help="skip the second, occluded run")
    p.add_argument("--out", help="write the markdown report here as well")
    args = p.parse_args(argv)
    cfg = load_config()
    videos = repo_path("data/videos")
    clips = [videos / f"{c}.mp4" for c in args.clips] if args.clips else sorted(videos.glob("*.mp4"))
    results_dir = repo_path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    lines = [f"# Track IDs ({date.today()})", "",
             f"Every frame; detector and tracker from config.yaml (max_lost_s = {cfg['track'].get('max_lost_s')}). "
             f"Occlusion test: detections suppressed for {OCCLUDE_S} s starting "
             f"{', '.join(f'{a:.0%}' for a in OCCLUDE_AT)} into the clip; 'back' = IDs that returned / IDs before, "
             f"with how far the faces moved meanwhile in face heights.", "",
             "| clip | people | ids | ids >2s | jumps | longest gap bridged | occlusion: back (moved) | ok? |",
             "|---|---|---|---|---|---|---|---|"]
    t_start = time.perf_counter()
    for clip in clips:
        people = PEOPLE.get(clip.stem.split("_")[0], "?")
        writer, scale = None, 1.0
        if args.video:
            cap = cv2.VideoCapture(str(clip))
            w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
            cap.release()
            scale = min(1.0, 960 / w)
            size = (int(w * scale), int(h * scale)) if scale < 1 else (w, h)
            writer = cv2.VideoWriter(str(results_dir / f"{clip.stem}_tracks.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, size)
        dets_all, times, fps, (w, h) = detect_clip(clip, cfg, writer, scale, size if args.video else None)
        if writer is not None:
            writer.release()
        n = len(dets_all)
        frames = track_clip(dets_all, times, fps, cfg)
        a = analyse(frames, fps, w)
        occ, occ_ok = "", True
        if not args.no_occlusion:
            cells = []
            for at in OCCLUDE_AT:
                start = int(n * at)
                window = (start, start + int(OCCLUDE_S * fps))
                occ_frames = track_clip(dets_all, times, fps, cfg, window)
                nb, back, moved = occlusion_result(occ_frames, frames, window)
                cells.append(f"{back}/{nb} ({moved:.1f}h)" if nb else "-")
                if nb and back < nb and moved < 1.0:      # stayed roughly put and still lost the ID
                    occ_ok = False
            occ = ", ".join(cells)
        ok = "yes" if (people == "?" or a["long_ids"] <= people) and not a["jumps"] and occ_ok else "look"
        jumps = f"{len(a['jumps'])}" + (f" (at {', '.join(f'{t:.1f}s' for _, t, _ in a['jumps'][:6])}{'...' if len(a['jumps']) > 6 else ''})" if a["jumps"] else "")
        lines.append(f"| {clip.stem} | {people} | {a['ids']} | {a['long_ids']} | {jumps} | "
                     f"{a['gap_frames']} frames ({a['gap_frames'] / fps:.2f} s) | {occ or '-'} | {ok} |")
        print(lines[-1], flush=True)
    lines += ["", f"{(time.perf_counter() - t_start) / 60:.1f} min wall.", "",
              "ids > people means a track was lost for longer than max_lost_s and came back as a new ID "
              "(re-ID, Task 12, merges those) or the detector found something that is not one of the people. "
              "The podcast clips are edited: a cut can also put a different person where the last one was, and "
              "a motion tracker then keeps the old ID (two_person_podcast: 1 ID for 2 people). Only re-ID can "
              "tell them apart; the real camera does not cut."]
    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
