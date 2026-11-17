#!/usr/bin/env python3
"""import_dataset.py - bring public datasets into the project's clip + label layout.

  python tools/import_dataset.py ami --meeting ES2008a --download --views closeup4 corner --audio
  python tools/import_dataset.py ava --id 053oq2xB3oU --download --start 900 --duration 120
  python tools/import_dataset.py realvad --video path/to/downloaded.mp4 --duration 300

Each writes data/videos/<name>.mp4 and its truth in data/labels/, in clip time (frame / fps),
so eval/speaking_accuracy.py and the rest of the pipeline treat it like any other clip. Raw
downloads stay in data/external/ (not committed).

ami      AMI Meeting Corpus (CC BY 4.0): four people in a meeting room. A close-up camera films
         one participant, so each close-up becomes a one-person clip whose label file says when
         that person speaks, taken from the word timings of the manual transcript (words closer
         than 0.5 s are one stretch). The wide views (corner, overhead) get all four speakers as
         persons 1-4 = A-D, which is NOT left-to-right order: check before scoring them.
         --audio also writes data/audio/<name>_mix.wav (headset mix) and _array1.wav (the 8
         channels of the table's circular microphone array, 10 cm radius) for the audio stage.
ava      AVA-ActiveSpeaker (labels CC BY 4.0; the videos are films from YouTube, mirrored for
         research): every face is boxed and labeled per frame. Becomes <name>.faces.csv, with
         speaking = SPEAKING_AUDIBLE or SPEAKING_NOT_AUDIBLE (both are lips moving).
realvad  RealVAD (CC BY 4.0): a panel of nine filmed by one fixed camera, with an upper-body box
         and a speaking flag per panelist per frame. The video is not in the dataset: its readme
         says to download www.youtube.com/watch?v=51pRTOIso4U yourself and pass it as --video.
         Not yet tested against the real video: look at the check image this writes.

Clips are re-encoded with OpenCV (video only), so they can be cut with --start / --duration.
"""

from __future__ import annotations

import argparse
import csv
import sys
import tarfile
import urllib.request
import wave
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv_env  # noqa: F401, E402
import cv2  # noqa: E402

from config import repo_path  # noqa: E402
from labels import FaceLabel, FaceLabels, Labels, Segment  # noqa: E402

AMI_MEDIA = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
AMI_ANNOTATIONS = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
AVA_LABELS = "https://research.google.com/ava/download/ava_activespeaker_{split}_v1.0.tar.bz2"
AVA_NAMES = "https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt"
AVA_VIDEOS = "https://s3.amazonaws.com/ava-dataset/trainval"
MERGE_GAP_S = 0.5              # words closer than this are one stretch of speech (the rule in data/README.md)


# --- shared ---------------------------------------------------------------------------------
def data_dir(args) -> Path:
    """Where clips, labels and audio are written: data/ unless --data-dir says otherwise."""
    return repo_path(args.data_dir)


def fetch(url: str, path: Path) -> Path:
    """Download unless the file is already there."""
    if path.is_file() and path.stat().st_size > 0:
        print(f"have     {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {url}\n      -> {path}")
    tmp = path.with_name(path.name + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return path


def trim_video(src: Path, dst: Path, start: float = 0.0, duration: float | None = None):
    """Re-encode [start, start + duration) of src to dst (mp4v, no audio).

    Returns (fps, width, height, frames written, the start actually used in seconds): frame i of
    dst is frame skip + i of src, so a source time u becomes u - skip / fps in the clip.
    """
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    skip = int(round(start * fps))
    for _ in range(skip):                         # decode forward rather than seek: exact on any container
        if not cap.grab():
            break
    limit = None if duration is None else int(round(duration * fps))
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while limit is None or n < limit:
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)
        n += 1
    out.release()
    cap.release()
    return fps, w, h, n, skip / fps


def merge_stretches(times: list[tuple[float, float]], gap: float = MERGE_GAP_S) -> list[tuple[float, float]]:
    """Sorted (start, end) pairs -> stretches, joining pairs closer than `gap`."""
    out: list[list[float]] = []
    for a, b in sorted(times):
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def cut(stretches, start: float, end: float | None) -> list[tuple[float, float]]:
    """Keep the part of each stretch inside [start, end) and shift it to clip time."""
    out = []
    for a, b in stretches:
        a2, b2 = max(a, start), b if end is None else min(b, end)
        if b2 > a2:
            out.append((a2 - start, b2 - start))
    return out


# --- AMI ------------------------------------------------------------------------------------
def ami_speakers(meetings_xml: Path, meeting: str) -> dict[str, dict]:
    """agent letter -> {camera, channel, name, role} for one meeting."""
    for m in ET.parse(meetings_xml).getroot().iter("meeting"):
        if m.get("observation") == meeting:
            return {s.get("nxt_agent"): dict(camera=s.get("camera"), channel=s.get("channel"),
                                             name=s.get("global_name"), role=s.get("role"))
                    for s in m.iter("speaker")}
    raise KeyError(f"meeting {meeting} is not in {meetings_xml}")


def ami_word_times(words_xml: Path) -> list[tuple[float, float]]:
    """(start, end) of every spoken word: no punctuation, laughs, coughs or gaps."""
    out = []
    for w in ET.parse(words_xml).getroot().iter("w"):
        if w.get("punc") == "true" or w.get("starttime") is None or w.get("endtime") is None:
            continue
        a, b = float(w.get("starttime")), float(w.get("endtime"))
        if b > a:
            out.append((a, b))
    return out


def ami_wav_stack(paths: list[Path], dst: Path, start: float, duration: float | None) -> None:
    """Cut [start, start + duration) from mono 16-bit wavs and write them as one multi-channel wav."""
    chans, rate = [], None
    for p in paths:
        with wave.open(str(p), "rb") as w:
            if w.getsampwidth() != 2 or w.getnchannels() != 1:
                raise RuntimeError(f"{p.name}: expected mono 16-bit PCM")
            rate = w.getframerate()
            w.setpos(min(w.getnframes(), int(round(start * rate))))
            n = w.getnframes() - w.tell() if duration is None else int(round(duration * rate))
            chans.append(np.frombuffer(w.readframes(n), dtype="<i2"))
    n = min(len(c) for c in chans)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(len(chans))
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.stack([c[:n] for c in chans], axis=1).astype("<i2").tobytes())
    print(f"wrote {dst} ({len(chans)} ch, {rate} Hz, {n / rate:.1f} s)")


def cmd_ami(args) -> int:
    src = repo_path(args.src)
    ann, media = src / "annotations", src / args.meeting
    if args.download:
        if not (ann / "corpusResources" / "meetings.xml").is_file():
            with zipfile.ZipFile(fetch(AMI_ANNOTATIONS, src / "ami_public_manual_1.6.2.zip")) as z:
                z.extractall(ann)
    speakers = ami_speakers(ann / "corpusResources" / "meetings.xml", args.meeting)
    stretches = {agent: merge_stretches(ami_word_times(ann / "words" / f"{args.meeting}.{agent}.words.xml"))
                 for agent in sorted(speakers)}
    end = None if args.duration is None else args.start + args.duration
    for view in args.views:
        stream = view.capitalize() if not view.lower().startswith("closeup") else "Closeup" + view[-1]
        name = f"ami_{args.meeting.lower()}_{view.lower()}"
        avi = media / f"{args.meeting}.{stream}.avi"
        if args.download:
            fetch(f"{AMI_MEDIA}/{args.meeting}/video/{avi.name}", avi)
        if not avi.is_file():
            print(f"skip {view}: {avi} is missing (use --download)", file=sys.stderr)
            continue
        fps, w, h, n, t0 = trim_video(avi, data_dir(args) / "videos" / f"{name}.mp4", args.start, args.duration)
        if stream.startswith("Closeup"):
            agents = [a for a, s in speakers.items() if s["camera"] == stream]
            who = {a: 1 for a in agents}
        else:
            who = {a: i + 1 for i, a in enumerate(sorted(speakers))}
        segs = [Segment(a, b, person) for agent, person in who.items() for a, b in cut(stretches[agent], t0, end)]
        Labels(segs).save(data_dir(args) / "labels" / f"{name}.csv")
        talk = sum(s.t_end - s.t_start for s in segs)
        print(f"wrote data/videos/{name}.mp4 ({w}x{h}, {fps:.2f} fps, {n / fps:.1f} s) and data/labels/{name}.csv: "
              f"{len(segs)} stretches, {talk:.0f} s of speech, " +
              ", ".join(f"person {p} = speaker {a} ({speakers[a]['name']}, {speakers[a]['role']})" for a, p in who.items()))
        if not stream.startswith("Closeup"):
            print("  note: persons 1-4 are speakers A-D, not left-to-right order; check before scoring this view")
    if args.audio:
        base = f"ami_{args.meeting.lower()}"
        wanted = {"mix": [f"{args.meeting}.Mix-Headset.wav"],
                  "array1": [f"{args.meeting}.Array1-0{i}.wav" for i in range(1, 9)]}
        for tag, names in wanted.items():
            paths = [media / n for n in names]
            if args.download:
                for p in paths:
                    fetch(f"{AMI_MEDIA}/{args.meeting}/audio/{p.name}", p)
            if all(p.is_file() for p in paths):
                ami_wav_stack(paths, data_dir(args) / "audio" / f"{base}_{tag}.wav", args.start, args.duration)
            else:
                print(f"skip audio {tag}: files missing in {media} (use --download)", file=sys.stderr)
    return 0


# --- AVA-ActiveSpeaker ----------------------------------------------------------------------
def ava_rows(label_csv: Path):
    """[(timestamp, x1, y1, x2, y2 (0..1), label, entity_id)] from one AVA active-speaker file."""
    with open(label_csv, newline="", encoding="utf-8") as fh:
        return [(float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), r[6], r[7])
                for r in csv.reader(fh) if len(r) >= 8 and r[1].replace(".", "", 1).isdigit()]


def ava_to_faces(rows, w: int, h: int, t0: float, end: float | None) -> FaceLabels:
    ids: dict[str, int] = {}
    out = []
    for ts, x1, y1, x2, y2, label, entity in sorted(rows):
        if ts < t0 or (end is not None and ts >= end):
            continue
        person = ids.setdefault(entity, len(ids) + 1)
        out.append(FaceLabel(ts - t0, x1 * w, y1 * h, x2 * w, y2 * h, label.startswith("SPEAKING"), person, "face"))
    return FaceLabels(out)


def cmd_ava(args) -> int:
    src = repo_path(args.src)
    label_csv = Path(args.labels) if args.labels else None
    video = Path(args.video) if args.video else None
    if args.download:
        for split in ("val", "train"):
            if label_csv is None or not label_csv.is_file():
                tar = fetch(AVA_LABELS.format(split=split), src / f"ava_activespeaker_{split}_v1.0.tar.bz2")
                with tarfile.open(tar) as t:
                    member = next((m for m in t.getmembers() if Path(m.name).name == f"{args.id}-activespeaker.csv"), None)
                    if member is not None:
                        label_csv = src / "labels" / Path(member.name).name
                        label_csv.parent.mkdir(parents=True, exist_ok=True)
                        label_csv.write_bytes(t.extractfile(member).read())
        names = fetch(AVA_NAMES, src / "ava_file_names_trainval_v2.1.txt").read_text().split()
        fname = next((n for n in names if n.rsplit(".", 1)[0] == args.id), None)
        if fname is None:
            print(f"{args.id} is not on the video mirror", file=sys.stderr)
            return 1
        video = fetch(f"{AVA_VIDEOS}/{fname}", src / fname)
    if label_csv is None:
        found = sorted(src.rglob(f"{args.id}-activespeaker.csv"))
        label_csv = found[0] if found else None
    if video is None:
        found = [p for p in sorted(src.glob(f"{args.id}.*")) if p.suffix in (".mkv", ".mp4", ".webm")]
        video = found[0] if found else None
    if not label_csv or not label_csv.is_file() or not video or not video.is_file():
        print(f"need the label file and the video for {args.id}: pass --labels/--video or --download", file=sys.stderr)
        return 1
    rows = ava_rows(label_csv)
    first, last = min(r[0] for r in rows), max(r[0] for r in rows)
    start = first if args.start is None else args.start
    name = args.name or f"ava_{args.id.lower().replace('-', '_')}_{int(start)}s"
    fps, w, h, n, t0 = trim_video(video, data_dir(args) / "videos" / f"{name}.mp4", start, args.duration)
    faces = ava_to_faces(rows, w, h, t0, t0 + n / fps)
    faces.save(data_dir(args) / "labels" / f"{name}.faces.csv")
    spoke = sum(r.speaking for r in faces.rows)
    print(f"labels cover {first:.0f}-{last:.0f} s of the film. wrote data/videos/{name}.mp4 ({w}x{h}, {fps:.2f} fps, "
          f"{n / fps:.1f} s from {t0:.1f} s) and data/labels/{name}.faces.csv: {len(faces.rows)} face labels, "
          f"{len(faces.people())} face tracks, {100 * spoke / max(len(faces.rows), 1):.0f}% speaking")
    return 0


# --- RealVAD --------------------------------------------------------------------------------
REALVAD_SIZE = (1280, 720)        # the frame the published boxes fit; they are scaled to the video given


def realvad_panelist(folder: Path, k: int):
    """({frame: (x1, y1, x2, y2)}, {frame: speaking}) for panelist k.

    The dataset's readme calls the last two box numbers width and height, but as such the boxes
    reach x = 2277; as a second corner (x2, y2) they all fit a 1280x720 frame. Rows repeat and
    jump back in some files, so the last row for a frame wins.
    """
    boxes = {int(r[0]): tuple(float(v) for v in r[1:5])
             for r in np.loadtxt(folder / f"Panelist{k}_bbox.txt", ndmin=2)}
    vad = {int(r[0]): bool(r[1]) for r in np.loadtxt(folder / f"Panelist{k}_VAD.txt", ndmin=2)}
    return boxes, vad


def realvad_to_faces(folder: Path, fps: float, w: int, h: int, t0: float, end: float | None) -> FaceLabels:
    sx, sy = w / REALVAD_SIZE[0], h / REALVAD_SIZE[1]
    out = []
    for k in range(1, 10):
        if not (folder / f"Panelist{k}_VAD.txt").is_file():
            continue
        boxes, vad = realvad_panelist(folder, k)
        for frame, speaking in vad.items():
            t = frame / fps
            if frame not in boxes or t < t0 or (end is not None and t >= end):
                continue
            x1, y1, x2, y2 = boxes[frame]
            if x2 > x1 and y2 > y1:
                out.append(FaceLabel(t - t0, x1 * sx, y1 * sy, x2 * sx, y2 * sy, speaking, k, "body"))
    return FaceLabels(out)


def cmd_realvad(args) -> int:
    folder, video = repo_path(args.src), Path(args.video)
    if not video.is_file():
        print(f"{video} not found. The dataset has no video: download www.youtube.com/watch?v=51pRTOIso4U "
              "and pass the file as --video", file=sys.stderr)
        return 1
    name = args.name or "realvad_panel" + (f"_{int(args.start)}s" if args.start else "")
    fps, w, h, n, t0 = trim_video(video, data_dir(args) / "videos" / f"{name}.mp4", args.start, args.duration)
    if abs(w / h - REALVAD_SIZE[0] / REALVAD_SIZE[1]) > 0.02:
        print(f"warning: {w}x{h} is not 16:9 like the labeled video; the boxes may not line up", file=sys.stderr)
    faces = realvad_to_faces(folder, fps, w, h, t0, t0 + n / fps)
    faces.save(data_dir(args) / "labels" / f"{name}.faces.csv")
    check = data_dir(args) / "videos" / f"{name}_check.jpg"
    cap = cv2.VideoCapture(str(data_dir(args) / "videos" / f"{name}.mp4"))
    mid = faces.span()[0] + 0.5 * (faces.span()[1] - faces.span()[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid * fps))
    ok, frame = cap.read()
    cap.release()
    if ok:
        for r in faces.at(mid, tol=1.0):
            color = (60, 200, 60) if r.speaking else (190, 190, 190)
            cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), color, 2)
            cv2.putText(frame, f"{r.person}", (int(r.x1) + 4, int(r.y1) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imwrite(str(check), frame)
    print(f"wrote data/videos/{name}.mp4 ({w}x{h}, {fps:.2f} fps, {n / fps:.1f} s) and data/labels/{name}.faces.csv: "
          f"{len(faces.rows)} labels for {len(faces.people())} panelists.\nLOOK AT {check}: each box should sit on one "
          "panelist's upper body. If they do not, the frame rate or size of this video differs from the labeled one.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="dataset", required=True)
    a = sub.add_parser("ami", help="AMI Meeting Corpus: close-up clips with per-speaker speech times, array audio")
    a.add_argument("--meeting", required=True, help="meeting id, e.g. ES2008a")
    a.add_argument("--views", nargs="+", default=["closeup1", "closeup2", "closeup3", "closeup4"],
                   help="closeup1..4, corner, overhead (default: the four close-ups)")
    a.add_argument("--audio", action="store_true", help="also write the headset mix and the 8-channel array to data/audio/")
    a.add_argument("--src", default="data/external/ami")
    a.set_defaults(func=cmd_ami)
    v = sub.add_parser("ava", help="AVA-ActiveSpeaker: one film with per-face speaking labels")
    v.add_argument("--id", required=True, help="video id, e.g. 053oq2xB3oU")
    v.add_argument("--video", help="the downloaded film (default: data/external/ava/<id>.*)")
    v.add_argument("--labels", help="<id>-activespeaker.csv (default: found under data/external/ava)")
    v.add_argument("--name", help="clip name (default: ava_<id>_<start>s)")
    v.add_argument("--src", default="data/external/ava")
    v.set_defaults(func=cmd_ava)
    r = sub.add_parser("realvad", help="RealVAD: nine-person panel, fixed camera, speaking flag per panelist")
    r.add_argument("--video", required=True, help="the panel video downloaded from YouTube (see the readme)")
    r.add_argument("--name", help="clip name (default: realvad_panel)")
    r.add_argument("--src", default="data/external/realvad/OurDataset_Public")
    r.set_defaults(func=cmd_realvad)
    for s in (a, v, r):
        s.add_argument("--start", type=float, default=None if s is v else 0.0,
                       help="cut from this second of the source" + (" (default: where the labels start)" if s is v else ""))
        s.add_argument("--duration", type=float, help="cut this many seconds (default: to the end)")
        s.add_argument("--data-dir", default="data", help="write videos/, labels/, audio/ under this folder (default data)")
        if s is not r:
            s.add_argument("--download", action="store_true", help="fetch what is missing into data/external/")
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
