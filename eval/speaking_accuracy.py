#!/usr/bin/env python3
"""speaking_accuracy.py - Task 8 check: the lip-motion speaking score against who is really speaking.

  python eval/speaking_accuracy.py --extract      # slow, once per clip: mouth ratios -> output/speaking/<clip>.csv
  python eval/speaking_accuracy.py                # fast: score, compare, write eval/results/speaking.md
  python eval/speaking_accuracy.py --tune         # compare methods, windows and thresholds
  python eval/speaking_accuracy.py --clips two_person_overlap --threshold 0.025

The slow part (detector, tracker, landmarks) is cached, so thresholds can be tried in seconds.
Scoring also writes output/speaking/<clip>_scores.csv (one row per face per frame: ratio,
fluctuation, probability, prediction, truth) for eval/speaking_threshold.ipynb.

Ground truth per clip, best first:
  faces    data/labels/<clip>.faces.csv (imported datasets): a box and a speaking flag per
           person per frame. Each tracked face takes the truth of the labeled box it is in;
           faces in no labeled box are not judged.
  labels   data/labels/<clip>.csv from tools/label.py; a header-only file means nobody speaks.
           Person k is the k-th face from the left in the first frame that shows everyone;
           each track belongs to the person whose position its median position is nearest, and
           a face far from everyone's place (a passer-by) is not judged.
  audio    no label file but an audible sound track: speech / no speech from the audio level.
           WEAK: it says someone is speaking, not who, so the frame counts as "speaking" if any
           visible face is predicted speaking, and a listener shown while someone off screen
           talks counts against the score.
Clips with neither are listed with their score statistics only.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config, repo_path  # noqa: E402
from labels import FaceLabels, Labels  # noqa: E402
from speaking import LipMotion, SpeakingState, motion_args, probability  # noqa: E402

CACHE = "output/speaking"
TARGET = 0.85


# --- the slow pass -------------------------------------------------------------------------
def extract(clips: list[Path], cfg: dict, cache: Path) -> None:
    from av_event_log import SessionClock
    from capture import VideoSource
    from detect import FaceDetector
    from speaking import SpeakingScorer
    from track import FaceTracker

    detector, scorer = FaceDetector(cfg["detect"]), SpeakingScorer(cfg["speaking"])
    cache.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        n = faces = found = 0
        with VideoSource(str(clip), cfg["camera"], SessionClock()) as video, \
                open(cache / f"{clip.stem}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "track_id", "cx", "cy", "h", "ratio", "frame_w", "frame_h", "yaw", "pitch"])
            tracker = FaceTracker(cfg["track"], fps=video.fps)
            for t, frame in video:
                n += 1
                tracks = tracker.update(detector.detect(frame), frame, t)
                if not tracks:
                    w.writerow([f"{t:.4f}", "", "", "", "", "", video.width, video.height, "", ""])
                for trk in tracks:
                    r, yaw, pitch = scorer.measure(trk, frame)      # raw: the gates are applied in replay()
                    faces += 1
                    found += r is not None
                    w.writerow([f"{t:.4f}", trk.track_id, f"{trk.cx:.1f}", f"{trk.cy:.1f}", f"{trk.height:.1f}",
                                "" if r is None else f"{r:.5f}", video.width, video.height,
                                "" if yaw is None else f"{yaw:.3f}", "" if pitch is None else f"{pitch:.3f}"])
        print(f"extracted {clip.stem}: {n} frames, {faces} face-frames, landmarks on {100 * found / max(faces, 1):.0f}%", flush=True)


def load_rows(path: Path):
    """[(t, [(track_id, cx, ratio or None, cy, yaw, pitch), ...])] in time order, and the frame width."""
    frames, width = defaultdict(list), 0
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            t = float(r["t"])
            width = int(r["frame_w"])
            frames[t]                                            # a frame without faces still counts
            if r["track_id"]:
                num = lambda k: float(r[k]) if r.get(k) else None     # noqa: E731  (older caches have no pose)
                frames[t].append((int(r["track_id"]), float(r["cx"]), num("ratio"), float(r["cy"]), num("yaw"), num("pitch")))
    return sorted(frames.items()), width


# --- the fast pass -------------------------------------------------------------------------
def replay(frames, sp: dict):
    """Run the cached ratios through LipMotion, with the scorer's gates.

    Returns rows (t, track_id, cx, ratio, fluctuation, cy, usable): usable is False where no
    probability is given (face turned past max_yaw, holes in the landmarks, head moving)."""
    motion, out = {}, []
    args, max_yaw = motion_args(sp), sp.get("max_yaw")
    for t, faces in frames:
        for face in faces:
            tid, cx, ratio = face[:3]
            cy, yaw, pitch = (list(face[3:6]) + [None] * 3)[:3]
            if max_yaw is not None and yaw is not None and abs(yaw) > max_yaw:
                ratio = None
            m = motion.setdefault(tid, LipMotion(**args))
            m.hold(t) if ratio is None else m.update(t, ratio, yaw, pitch)
            usable = ratio is not None and (args["max_head_motion"] is None or m.head_motion <= args["max_head_motion"])
            out.append((t, tid, cx, ratio, m.variability if usable else 0.0, cy, usable))
    return out


def decide(rows, sp: dict):
    """Probability and on/off state (hysteresis) per row from the fluctuation values."""
    state, by_t = SpeakingState(sp), defaultdict(list)
    for i, row in enumerate(rows):
        usable = row[6] if len(row) > 6 else True
        by_t[row[0]].append((i, row[1], probability(row[4], sp["threshold"], sp["slope"]) if usable else 0.0))
    prob, pred = [0.0] * len(rows), [False] * len(rows)
    for t in sorted(by_t):
        on = state.update({tid: p for _, tid, p in by_t[t]})
        for i, tid, p in by_t[t]:
            prob[i], pred[i] = p, tid in on
    return prob, pred


def audio_activity(clip: Path):
    """speech(t) -> bool from the sound track's level, or None when there is no audible audio."""
    try:
        import av
    except ImportError:
        return None
    with av.open(str(clip)) as c:
        if not c.streams.audio:
            return None
        rs = av.AudioResampler(format="flt", layout="mono", rate=16000)
        x = np.concatenate([f.to_ndarray().reshape(-1) for fr in c.decode(c.streams.audio[0]) for f in rs.resample(fr)])
    hop = 800                                                    # 50 ms
    fr = x[:len(x) // hop * hop].reshape(-1, hop)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)
    if np.percentile(db, 90) < -40:                              # a silent or near-silent track
        return None
    active = db > max(-35.0, np.percentile(db, 10) + 10)
    k = 3                                                        # bridge gaps up to 150 ms each side
    padded = np.convolve(active.astype(float), np.ones(2 * k + 1), mode="same") > 0
    return lambda t: bool(padded[min(len(padded) - 1, max(0, int(t / 0.05)))])


def person_of_track(rows, n_people: int, width: float | None = None) -> dict[int, int]:
    """Track id -> person number: anchors are the faces, left to right, in the first frame showing n_people.

    With the frame width given, a track further than a fifth of it from every anchor belongs to nobody
    (someone walking through the shot, a person at the whiteboard) and is not judged.
    """
    by_t = defaultdict(list)
    for t, tid, cx, *_ in rows:
        by_t[t].append(cx)
    anchors = next((sorted(cxs) for t, cxs in sorted(by_t.items()) if len(cxs) == n_people), None)
    if anchors is None:
        return {}
    cxs = defaultdict(list)
    for t, tid, cx, *_ in rows:
        cxs[tid].append(cx)
    out = {}
    for tid, v in cxs.items():
        d = [abs(np.median(v) - a) for a in anchors]
        if width is None or min(d) <= 0.2 * width:
            out[tid] = int(np.argmin(d)) + 1
    return out


def truth_for(clip: Path, rows, label_dir: Path, use_audio: bool = True, width: float | None = None):
    """(source, truth per row or None, n_frames-level flag). truth[i] is True/False, or None = not judged."""
    face_file, label_file = label_dir / f"{clip.stem}.faces.csv", label_dir / f"{clip.stem}.csv"
    if face_file.is_file():
        fl = FaceLabels.load(face_file)
        return "face labels", [None if r[5] is None else fl.speaking_for(r[0], r[2], r[5]) for r in rows], False
    if label_file.is_file():
        lb = Labels.load(label_file)
        people = max(lb.people(), default=0)
        if people == 0:
            return "labels (no speech)", [False] * len(rows), False
        who = person_of_track(rows, people, width)
        return "labels", [(who[tid] in lb.speaking_at(t)) if tid in who else None for t, tid, *_ in rows], False
    speech = audio_activity(clip) if use_audio and clip.is_file() else None
    if speech is not None:
        return "audio (weak)", [speech(t) for t, *_ in rows], True
    return "none", None, False


def confusion(pred, truth, rows, any_face: bool):
    """tp, fp, fn, tn over face-frames, or over frames (any face predicted) for the weak audio truth."""
    if any_face:
        by_t = defaultdict(lambda: [False, False])
        for (t, *_), p, y in zip(rows, pred, truth):
            by_t[t][0] |= p
            by_t[t][1] = y
        pairs = list(by_t.values())
    else:
        pairs = [(p, y) for p, y in zip(pred, truth) if y is not None]
    tp = sum(p and y for p, y in pairs)
    fp = sum(p and not y for p, y in pairs)
    fn = sum((not p) and y for p, y in pairs)
    return tp, fp, fn, len(pairs) - tp - fp - fn


def scores(tp, fp, fn, tn) -> dict:
    n = tp + fp + fn + tn
    return dict(n=n, accuracy=(tp + tn) / n if n else float("nan"),
                precision=tp / (tp + fp) if tp + fp else float("nan"),
                recall=tp / (tp + fn) if tp + fn else float("nan"),
                false_positive_rate=fp / (fp + tn) if fp + tn else float("nan"))


def auc(values, truth) -> float:
    """Chance that a speaking sample has a higher fluctuation than a silent one."""
    v, y = np.asarray(values, float), np.asarray(truth, bool)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v))
    ranks[order] = np.arange(1, len(v) + 1)
    for val in np.unique(v[np.concatenate(([False], np.diff(v[order]) == 0))]):   # average the ranks of ties
        tie = v == val
        ranks[tie] = ranks[tie].mean()
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def pct(x) -> str:
    return "-" if x != x else f"{100 * x:.0f}%"


# --- command line --------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--clips", nargs="*", help="clip names in data/videos (default: all)")
    p.add_argument("--extract", action="store_true", help="run the slow pass and cache mouth ratios")
    p.add_argument("--tune", action="store_true", help="compare methods, windows and thresholds on the judged clips")
    p.add_argument("--threshold", type=float, help="override speaking.threshold")
    p.add_argument("--window", type=float, help="override speaking.window_s")
    p.add_argument("--method", choices=["std", "band"], help="override speaking.method")
    p.add_argument("--max-yaw", type=float, help="override speaking.max_yaw (0 = no limit)")
    p.add_argument("--max-head-motion", type=float, help="override speaking.max_head_motion (0 = no limit)")
    p.add_argument("--no-audio", action="store_true", help="do not use audio as weak truth")
    p.add_argument("--out", default="eval/results/speaking.md", help="markdown report (default %(default)s)")
    args = p.parse_args(argv)

    cfg = load_config()
    sp = dict(cfg["speaking"])
    for key, val in (("threshold", args.threshold), ("window_s", args.window), ("method", args.method)):
        if val is not None:
            sp[key] = val
    for key, val in (("max_yaw", args.max_yaw), ("max_head_motion", args.max_head_motion)):
        if val is not None:
            sp[key] = val or None
    videos, cache, label_dir = repo_path("data/videos"), repo_path(CACHE), repo_path("data/labels")
    clips = [videos / f"{c}.mp4" for c in args.clips] if args.clips else sorted(videos.glob("*.mp4"))
    if args.extract:
        extract(clips, cfg, cache)
        return 0
    clips = [c for c in clips if (cache / f"{c.stem}.csv").is_file()]
    if not clips:
        print(f"no cached ratios in {cache}: run with --extract first", file=sys.stderr)
        return 1

    data = {}
    for clip in clips:
        frames, width = load_rows(cache / f"{clip.stem}.csv")
        rows = replay(frames, sp)
        source, truth, any_face = truth_for(clip, rows, label_dir, not args.no_audio, width)
        data[clip.stem] = (frames, rows, source, truth, any_face)

    if args.tune:
        return tune(data, sp)

    lines = [f"# Lip-motion speaking score ({date.today()})", "",
             f"method `{sp['method']}`, window {sp['window_s']} s, threshold {sp['threshold']}, slope {sp['slope']}, "
             f"on/off {sp['on_threshold']}/{sp['off_threshold']}. Target: {TARGET:.0%} frame-level accuracy on labeled clips.", "",
             "| clip | truth | judged | accuracy | precision | recall | false-positive rate | fluctuation p50 / p90 |",
             "|---|---|---|---|---|---|---|---|"]
    for name, (frames, rows, source, truth, any_face) in data.items():
        prob, pred = decide(rows, sp)
        v = [r[4] for r in rows]
        spread = f"{np.percentile(v, 50):.3f} / {np.percentile(v, 90):.3f}" if v else "-"
        if truth is None:
            lines.append(f"| {name} | none | - | - | - | - | - | {spread} |")
        else:
            s = scores(*confusion(pred, truth, rows, any_face))
            unit = "frames" if any_face else "face-frames"
            lines.append(f"| {name} | {source} | {s['n']} {unit} | {pct(s['accuracy'])} | {pct(s['precision'])} | "
                         f"{pct(s['recall'])} | {pct(s['false_positive_rate'])} | {spread} |")
        print(lines[-1])
        with open(cache / f"{name}_scores.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "track_id", "cx", "ratio", "fluctuation", "prob", "pred", "truth"])
            for i, (t, tid, cx, ratio, var, *_rest) in enumerate(rows):
                w.writerow([f"{t:.4f}", tid, f"{cx:.1f}", "" if ratio is None else f"{ratio:.5f}", f"{var:.5f}",
                            f"{prob[i]:.4f}", int(pred[i]), "" if truth is None or truth[i] is None else int(truth[i])])
    lines += ["", "Truth `labels` = data/labels/<clip>.csv (per person). `audio (weak)` = speech / no speech from the "
                  "sound track; it cannot say who speaks, so a frame is right when *any* visible face is predicted "
                  "speaking while there is speech. Lip motion cannot tell talking from chewing or mouthing words; "
                  "the microphones settle that later."]
    out = repo_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


def tune(data: dict, sp: dict) -> int:
    judged = {k: v for k, v in data.items() if v[3] is not None}
    if not judged:
        print("nothing to tune on: no clip has labels or audible audio", file=sys.stderr)
        return 1
    print(f"tuning on: {', '.join(f'{k} [{v[2]}]' for k, v in judged.items())}\n")
    print(f"{'method':6s} {'window':>6s} {'AUC':>6s} | best threshold -> mean accuracy over clips (min)")
    for method in ("std", "band"):
        for window in (0.5, 0.75, 1.0):
            trial = {**sp, "method": method, "window_s": window}
            per_clip, values, truths = {}, [], []
            for name, (frames, _, source, _, any_face) in judged.items():
                rows = replay(frames, trial)
                truth = data[name][3]
                per_clip[name] = (rows, truth, any_face)
                values += [r[4] for r, y in zip(rows, truth) if y is not None]
                truths += [y for y in truth if y is not None]
            best = None
            for thr in np.round(np.arange(0.006, 0.0601, 0.002), 3):
                t2 = {**trial, "threshold": float(thr), "slope": float(thr) / 4}
                accs = [scores(*confusion(decide(rows, t2)[1], truth, rows, any_face))["accuracy"]
                        for rows, truth, any_face in per_clip.values()]
                if best is None or np.mean(accs) > best[1]:
                    best = (float(thr), float(np.mean(accs)), float(np.min(accs)))
            print(f"{method:6s} {window:6.2f} {auc(values, truths):6.3f} | {best[0]:.3f} -> {100 * best[1]:.1f}% ({100 * best[2]:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
