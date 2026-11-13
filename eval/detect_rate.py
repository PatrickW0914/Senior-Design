#!/usr/bin/env python3
"""detect_rate.py - Task 4 check: how often each backend finds the faces, how big they are, how fast.

  python eval/detect_rate.py                                   # every clip in data/videos, every 5th frame
  python eval/detect_rate.py --clips one_person_walking one_person_turning --step 3
  python eval/detect_rate.py --backends scrfd mediapipe --out eval/results/detect_rate.md

There are no per-frame visibility labels, so the reference for "visible faces" is a strong,
generous detector: SCRFD det_10g (buffalo_l) at min_confidence 0.3 on the 1280-wide frame. A
reference face is "frontal" when the nose sits within 0.4 eye-distances of the midpoint between
the eyes (yaw under about 40 degrees). A backend box matches a reference face at IoU >= 0.4.

Per clip and backend: recall on all and on frontal reference faces, extra boxes per sampled frame
(boxes matching no reference face), face height percentiles at full resolution, median ms per
frame for the detector alone. The Task 4 target is 95% recall on frontal faces.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from av_event_log import SessionClock  # noqa: E402
from capture import VideoSource  # noqa: E402
from config import load_config, repo_path  # noqa: E402
from detect import Detection, FaceDetector, is_frontal  # noqa: E402

TARGET = 0.95
IOU_MATCH = 0.4


def presets(cfg: dict) -> dict[str, dict]:
    det = cfg["detect"]
    root = det.get("scrfd", {}).get("root", "models/insightface")
    return {
        "scrfd": {**det, "backend": "scrfd"},
        "scrfd-10g": {**det, "backend": "scrfd", "scrfd": {**det.get("scrfd", {}), "pack": "buffalo_l"}},
        "mediapipe": {**det, "backend": "mediapipe"},
        "reference": {"backend": "scrfd", "width": 1280, "min_confidence": 0.3,
                      "scrfd": {"pack": "buffalo_l", "det_size": 640, "root": root}},
    }


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def match(ref: list[Detection], dets: list[Detection]) -> tuple[list[bool], int]:
    """Greedy one-to-one matching. Returns (found flag per reference face, unmatched detections)."""
    found, used = [False] * len(ref), set()
    pairs = sorted(((iou(r.bbox, d.bbox), i, j) for i, r in enumerate(ref) for j, d in enumerate(dets)), reverse=True)
    for score, i, j in pairs:
        if score < IOU_MATCH:
            break
        if not found[i] and j not in used:
            found[i], used = True, used | {j}
    return found, len(dets) - len(used)


class Tally:
    def __init__(self):
        self.ref = self.hit = self.ref_frontal = self.hit_frontal = self.extra = self.frames = 0
        self.heights: list[float] = []
        self.ms: list[float] = []

    def add(self, ref, frontal, dets, ms):
        found, extra = match(ref, dets)
        self.frames += 1
        self.ref += len(ref)
        self.hit += sum(found)
        self.ref_frontal += sum(frontal)
        self.hit_frontal += sum(f and fr for f, fr in zip(found, frontal))
        self.extra += extra
        self.heights += [d.height for d in dets]
        self.ms.append(ms)

    def row(self) -> dict:
        rec = self.hit / self.ref if self.ref else float("nan")
        recf = self.hit_frontal / self.ref_frontal if self.ref_frontal else float("nan")
        h = np.percentile(self.heights, [10, 50, 90]) if self.heights else [float("nan")] * 3
        return dict(recall=rec, recall_frontal=recf, n_ref=self.ref, n_frontal=self.ref_frontal,
                    extra_per_frame=self.extra / max(self.frames, 1), h10=h[0], h50=h[1], h90=h[2],
                    ms=float(np.median(self.ms)) if self.ms else float("nan"))


def fmt(x, pct=False, digits=0):
    if x != x:
        return "-"
    return f"{100 * x:.{digits}f}%" if pct else f"{x:.{digits}f}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--clips", nargs="*", help="clip names in data/videos (default: all .mp4 there)")
    p.add_argument("--backends", nargs="*", default=["scrfd", "mediapipe"], help="presets: scrfd scrfd-10g mediapipe")
    p.add_argument("--step", type=int, default=5, help="evaluate every Nth frame (default 5)")
    p.add_argument("--out", help="write the markdown report here as well")
    args = p.parse_args(argv)

    cfg = load_config()
    pre = presets(cfg)
    videos = repo_path("data/videos")
    clips = [videos / f"{c}.mp4" for c in args.clips] if args.clips else sorted(videos.glob("*.mp4"))
    if not clips:
        print(f"no clips in {videos}", file=sys.stderr)
        return 1
    reference = FaceDetector(pre["reference"])
    detectors = {name: FaceDetector(pre[name]) for name in args.backends}

    lines = [f"# Face detection rate ({date.today()})", "",
             f"Every {args.step}th frame. Reference: SCRFD det_10g, 1280 px wide, confidence 0.3; frontal = "
             f"yaw ratio <= 0.4; match at IoU >= {IOU_MATCH}. Target: {TARGET:.0%} recall on frontal faces.", "",
             "| clip | backend | recall all | recall frontal | ref faces (frontal) | extra/frame | face h px p10/p50/p90 | ms/frame | 95%? |",
             "|---|---|---|---|---|---|---|---|---|"]
    totals = {name: Tally() for name in detectors}
    t_start = time.perf_counter()
    for clip in clips:
        tallies = {name: Tally() for name in detectors}
        with VideoSource(str(clip), cfg["camera"], SessionClock()) as video:
            for i, (t, frame) in enumerate(video):
                if i % args.step:
                    continue
                ref = reference.detect(frame)
                frontal = [is_frontal(r) for r in ref]
                for name, det in detectors.items():
                    dets = det.detect(frame)
                    tallies[name].add(ref, frontal, dets, det.last_ms)
                    totals[name].add(ref, frontal, dets, det.last_ms)
        for name, tally in tallies.items():
            r = tally.row()
            ok = "-" if r["n_frontal"] == 0 else ("yes" if r["recall_frontal"] >= TARGET else "NO")
            lines.append(f"| {clip.stem} | {name} | {fmt(r['recall'], True)} | {fmt(r['recall_frontal'], True)} | "
                         f"{r['n_ref']} ({r['n_frontal']}) | {fmt(r['extra_per_frame'], digits=2)} | "
                         f"{fmt(r['h10'])}/{fmt(r['h50'])}/{fmt(r['h90'])} | {fmt(r['ms'], digits=1)} | {ok} |")
            print(lines[-1], flush=True)
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, tally in totals.items():
        r = tally.row()
        ok = "yes" if r["recall_frontal"] >= TARGET else "NO"
        lines.append(f"| **all clips** | {name} | {fmt(r['recall'], True)} | {fmt(r['recall_frontal'], True)} | "
                     f"{r['n_ref']} ({r['n_frontal']}) | {fmt(r['extra_per_frame'], digits=2)} | "
                     f"{fmt(r['h10'])}/{fmt(r['h50'])}/{fmt(r['h90'])} | {fmt(r['ms'], digits=1)} | {ok} |")
        print(lines[-1])
    lines += ["", f"{sum(t.frames for t in totals.values()) // max(len(totals), 1)} frames per backend, "
                  f"{(time.perf_counter() - t_start) / 60:.1f} min wall.",
              "", "Extra boxes are detections that match no reference face: false positives, or faces the "
                  "reference missed. Face heights are at full resolution."]
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
