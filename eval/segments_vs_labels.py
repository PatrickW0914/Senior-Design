#!/usr/bin/env python3
"""segments_vs_labels.py - Task 9 check: do the pipeline's speaking segments match the labels?

  python eval/segments_vs_labels.py                 # every labeled clip with cached mouth ratios
  python eval/segments_vs_labels.py --clips ami_es2008a_closeup4 --gap 1.0 --release 0.5
  python eval/segments_vs_labels.py --transcripts output   # compare main.py's own <clip>_transcript.json

By default each clip is replayed from the cache that `eval/speaking_accuracy.py --extract` wrote
(mouth ratio and head pose per face per frame) through the real scorer and the pipeline exactly
as main.py builds it, so a clip takes seconds. The outputs land in output/segments/
(<clip>_events.jsonl, _transcript.json, _transcript.txt). With --transcripts DIR the transcripts
main.py wrote are compared instead (it also needs main.py's <clip>_faces.jsonl there).

Compared against data/labels/<clip>.csv (t_start, t_end, person), per clip:
  turns             labeled turns of at least 1 s. Shorter ones ("yeah", "okay") are below what a
                    score that looks back 0.75 s can resolve; they are counted but not judged.
  found             turns that some segment of that person overlaps
  pieces            segments per found turn: 1.0 is one segment per turn, more is a turn in pieces
  start / end       found turns whose start (end) has a segment start (end) within the tolerance,
                    and the median signed error (+ = the segment boundary is late)
  speech covered    labeled speaking time inside segments: of all of it, and of the part during
                    which the person's face was tracked and roughly frontal (the rest cannot be seen)
  segments right    segment time that is labeled speech
  extra             segments overlapping no labeled turn
A track belongs to the person whose seat it is nearest (see speaking_accuracy.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from av_event_log import FaceObs  # noqa: E402
from config import load_config, repo_path  # noqa: E402
from geometry import AzimuthModel  # noqa: E402
from labels import Labels  # noqa: E402
from main import build_pipeline  # noqa: E402
from speaking import SpeakingScorer  # noqa: E402
from speaking_accuracy import CACHE, load_rows, person_of_track  # noqa: E402

OUT = "output/segments"
MIN_TURN_S = 1.0
OVERRIDES = (("gap", "gap_merge_s"), ("onset", "asd_onset_s"), ("release", "asd_release_s"),
             ("tau", "tau_s"), ("min_duration", "min_duration_s"))


def observations(name: str, cfg: dict, cache: Path):
    """Cached measurements -> the scorer -> [(t, [FaceObs])], plus per track: positions, times seen, times frontal."""
    frames, width = load_rows(cache / f"{name}.csv")
    scorer = SpeakingScorer(cfg["speaking"])
    max_yaw = cfg["speaking"].get("max_yaw")
    angles = AzimuthModel.from_hfov(width, round(width * 9 / 16), float(cfg["camera"]["hfov_deg"]))
    out, cxs, seen, frontal = [], defaultdict(list), defaultdict(list), defaultdict(list)
    for t, faces in frames:
        obs = []
        for tid, cx, ratio, cy, yaw, pitch in faces:
            obs.append(FaceObs(tid, angles.azimuth(cx, cy), scorer.score_from(tid, t, ratio, yaw, pitch)))
            cxs[tid].append(cx)
            seen[tid].append(t)
            if max_yaw is None or yaw is None or abs(yaw) <= max_yaw:
                frontal[tid].append(t)
        out.append((t, obs))
    return out, cxs, width, seen, frontal


def segments(obs_frames, cfg: dict, out_prefix: str, overrides: dict | None = None):
    """Run the pipeline as main.py builds it over the observations. Returns [(track_id, t_start, t_end)]."""
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    pipe = build_pipeline(cfg, out_prefix, overrides)
    for t, obs in obs_frames:
        pipe.on_video_frame(t, obs)
    segs, _ = pipe.close()
    return [(s.track_id, s.t_start, s.t_end) for s in segs]


def from_transcript(name: str, folder: Path):
    """Segments and track positions from what main.py wrote."""
    payload = json.loads((folder / f"{name}_transcript.json").read_text(encoding="utf-8"))
    segs = [(s["track_id"], s["t_start"], s["t_end"]) for s in payload["segments"]]
    cxs, seen = defaultdict(list), defaultdict(list)
    for line in (folder / f"{name}_faces.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        for f in rec["faces"]:
            cxs[f["track_id"]].append(0.5 * (f["bbox"][0] + f["bbox"][2]))
            seen[f["track_id"]].append(rec["t"])
    return segs, cxs, None, seen, seen


def overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def time_in(intervals, others) -> float:
    return sum(overlap(a, b, c, d) for a, b in intervals for c, d in others)


def spans(times, gap: float = 0.5):
    """Stretches covered by frame times closer together than `gap`."""
    out = []
    for t in sorted(times):
        if out and t - out[-1][1] <= gap:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [(a, b) for a, b in out]


def compare(segs, cxs, width, seen, frontal, labels: Labels, tol: float) -> dict:
    people = max(labels.people(), default=0)
    rows = [(t, tid, cx) for tid in cxs for t, cx in zip(seen[tid], cxs[tid])]
    who = person_of_track(rows, people, width) if people else {}
    pred, vis = defaultdict(list), defaultdict(list)
    for tid, a, b in segs:
        if tid in who:
            pred[who[tid]].append((a, b))
    for tid, person in who.items():
        vis[person] += spans(frontal[tid])
    lab = defaultdict(list)
    for s in labels.sorted():
        lab[s.person].append((s.t_start, s.t_end))

    r = dict(turns=0, short=0, found=0, pieces=0, start_ok=0, end_ok=0, extra=0, lab_time=0.0, lab_frontal=0.0,
             covered=0.0, pred_time=0.0, pred_right=0.0, segments=len(segs))
    start_err, end_err = [], []
    for person in sorted(set(lab) | set(pred)):
        L, P = lab.get(person, []), pred.get(person, [])
        for a, b in L:
            if b - a < MIN_TURN_S:
                r["short"] += 1
                continue
            r["turns"] += 1
            over = [(c, d) for c, d in P if overlap(a, b, c, d) > 0]
            if over:
                r["found"] += 1
                r["pieces"] += len(over)
                ds = min((c - a for c, d in P), key=abs)
                de = min((d - b for c, d in P), key=abs)
                start_err.append(ds)
                end_err.append(de)
                r["start_ok"] += abs(ds) <= tol
                r["end_ok"] += abs(de) <= tol
        r["extra"] += sum(1 for c, d in P if not any(overlap(a, b, c, d) > 0 for a, b in L))
        r["lab_time"] += sum(b - a for a, b in L)
        r["lab_frontal"] += time_in(L, vis.get(person, []))
        r["covered"] += time_in(L, P)
        r["pred_time"] += sum(d - c for c, d in P)
        r["pred_right"] += time_in(P, L)
    if not people:                                   # a clip labeled "nobody speaks": every segment is extra
        r["extra"] = len(segs)
        r["pred_time"] = sum(b - a for _, a, b in segs)
    r["start_err"], r["end_err"] = start_err, end_err
    return r


def frac(a, b) -> str:
    return "-" if not b else f"{100 * a / b:.0f}%"


def med(xs) -> str:
    return "-" if not xs else f"{float(np.median(xs)):+.2f} s"


def row(name: str, r: dict) -> str:
    return (f"| {name} | {r['turns']} (+{r['short']} short) | {frac(r['found'], r['turns'])} | "
            f"{'-' if not r['found'] else format(r['pieces'] / r['found'], '.1f')} | "
            f"{frac(r['start_ok'], r['found'])} ({med(r['start_err'])}) | {frac(r['end_ok'], r['found'])} ({med(r['end_err'])}) | "
            f"{frac(r['covered'], r['lab_time'])} / {frac(r['covered'], r['lab_frontal'])} | "
            f"{frac(r['pred_right'], r['pred_time'])} | {r['segments']} | {r['extra']} |")


def add(total: dict, r: dict) -> None:
    for k, v in r.items():
        total[k] = total.get(k, [] if isinstance(v, list) else 0) + v


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--clips", nargs="*", help="clip names (default: every clip with a label file and a cache)")
    p.add_argument("--transcripts", help="compare the <clip>_transcript.json files in this folder instead of replaying")
    for flag, key in OVERRIDES:
        p.add_argument(f"--{flag.replace('_', '-')}", type=float, help=f"override fusion.{key}")
    p.add_argument("--tolerance", type=float, default=0.5, help="boundary tolerance in seconds (default 0.5)")
    p.add_argument("--out", default="eval/results/segments.md", help="markdown report (default %(default)s)")
    args = p.parse_args(argv)

    cfg = load_config()
    cache, label_dir, out_dir = repo_path(CACHE), repo_path("data/labels"), repo_path(OUT)
    overrides = {key: getattr(args, flag) for flag, key in OVERRIDES if getattr(args, flag) is not None}
    names = args.clips or sorted(f.stem for f in label_dir.glob("*.csv")
                                 if not f.name.endswith(".faces.csv") and (cache / f.name).is_file())
    if not names:
        print("no clip has both a label file and cached ratios (run eval/speaking_accuracy.py --extract)", file=sys.stderr)
        return 1
    fus = {**cfg["fusion"], **overrides}
    lines = [f"# Speaking segments against labels ({date.today()})", "",
             f"fusion mode `{fus.get('mode', 'audio_visual')}`, tau {fus.get('tau_s')} s, gap bridged {fus.get('gap_merge_s', 0.5)} s, "
             f"onset / release correction {fus.get('asd_onset_s', 0.0)} / {fus.get('asd_release_s', 0.0)} s; "
             f"boundary tolerance {args.tolerance} s; "
             + ("transcripts from " + args.transcripts if args.transcripts else "replayed from cached measurements") + ".", "",
             f"| clip | turns >= {MIN_TURN_S:g} s | found | pieces | start within tol (median error) | end within tol (median error) | "
             "speech covered: all / while frontal | segments right | segments | extra |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    total: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for name in names:
            labels = Labels.load(label_dir / f"{name}.csv")
            if args.transcripts:
                segs, cxs, width, seen, frontal = from_transcript(name, repo_path(args.transcripts))
            else:
                obs, cxs, width, seen, frontal = observations(name, cfg, cache)
                segs = segments(obs, cfg, str(out_dir / name), overrides)
            r = compare(segs, cxs, width, seen, frontal, labels, args.tolerance)
            add(total, r)
            lines.append(row(name, r))
            print(lines[-1], flush=True)
    lines.append(row("**all**", total))
    print(lines[-1])
    lines += ["", "\"While frontal\" counts only labeled speech during which that person's face was tracked and not turned "
                  "past speaking.max_yaw. Median errors are signed: positive means the segment boundary comes after the "
                  "labeled one. Turns under 1 s are listed as short and not judged."]
    out = repo_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
