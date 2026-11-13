#!/usr/bin/env python3
"""label.py - mark who is speaking in a clip (Task 3).

  python tools/label.py --source data/videos/two_person_overlap.mp4

Plays the clip with its audio. Press 1 (or 2, 3, ...) when that person starts speaking and
press it again when they stop; several people can be on at once. Writes
data/labels/<clip>.csv, one row per speaking segment: t_start,t_end,person. Times are
frame_index / fps, the time base capture.py gives the pipeline, so labels line up with
output/*_faces.jsonl. Running it again on the same clip continues the existing CSV.

Keys
  1-9      person starts / stops speaking      space    pause / resume
  j / l    back / forward 2 s                  , / .    one frame back / forward (pauses)
  - / =    half / double speed                 u        undo the last finished segment
  s        save                                q        save and quit (also closes open segments)

Press-to-start / press-to-stop rather than hold: a held key reaches cv2 as key-repeat events,
which stop for up to a second after the key goes down, so holding would lose segment starts.
A clip without usable audio plays silent and says so: label it from lip movement.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv_env  # noqa: F401, E402  (must stay above cv2)
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from av_event_log import SessionClock  # noqa: E402
from capture import VideoSource  # noqa: E402
from config import load_config, repo_path  # noqa: E402
from labels import Labels  # noqa: E402
from main import parse_source  # noqa: E402

SEEK_S = 2.0
SILENT_PEAK = 0.01          # a track whose loudest sample is below this is treated as silent
COLORS = [None, (0, 200, 255), (255, 140, 0), (80, 220, 80), (200, 80, 255),   # BGR, by person
          (255, 255, 0), (0, 120, 255), (255, 0, 160), (160, 160, 160), (0, 255, 255)]


class Audio:
    """The clip's audio decoded once (48 kHz stereo), played from any point with sounddevice."""

    RATE = 48000

    def __init__(self, data: np.ndarray):
        self.data = data

    @classmethod
    def load(cls, path: str) -> Audio | None:
        try:
            import av
            import sounddevice as sd
        except ImportError as e:   # both come with mediapipe; this is just in case
            print(f"audio off: {e}")
            return None
        try:
            with av.open(path) as container:
                if not container.streams.audio:
                    print("audio off: the clip has no audio track")
                    return None
                rs = av.AudioResampler(format="flt", layout="stereo", rate=cls.RATE)
                chunks = [f.to_ndarray().reshape(-1, 2)
                          for frame in container.decode(container.streams.audio[0])
                          for f in rs.resample(frame)]
                chunks += [f.to_ndarray().reshape(-1, 2) for f in rs.resample(None)]
            data = np.concatenate(chunks) if chunks else np.zeros((0, 2), np.float32)
            sd.check_output_settings(samplerate=cls.RATE, channels=2)
        except Exception as e:  # noqa: BLE001  - no sound device, odd codec, ...: label without audio
            print(f"audio off: {e}")
            return None
        peak = float(np.abs(data).max()) if len(data) else 0.0
        if peak < SILENT_PEAK:
            print(f"audio off: the track is silent (peak {peak:.3f})")
            return None
        return cls(data)

    def play_from(self, t: float, speed: float = 1.0) -> None:
        import sounddevice as sd
        try:
            sd.play(self.data[int(t * self.RATE):], int(self.RATE * speed))   # pitch follows speed
        except Exception as e:  # noqa: BLE001
            print(f"audio: {e}")

    def stop(self) -> None:
        import sounddevice as sd
        sd.stop()


def fit(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    return frame if w <= width else cv2.resize(frame, (width, int(h * width / w)))


def draw(frame, t, video, labels, people, playing, speed, audio_on, ended):
    img = frame.copy()
    H, W = img.shape[:2]
    f = cv2.FONT_HERSHEY_SIMPLEX
    state = "END" if ended else "PLAYING" if playing else "PAUSED"
    cv2.rectangle(img, (0, 0), (W, 58), (0, 0, 0), -1)
    cv2.putText(img, f"{t:7.2f} / {video.duration_s:.2f} s   x{speed:g}   {state}   "
                     f"audio {'on' if audio_on else 'OFF - label from lips'}   "
                     f"segments {len(labels.segments)}", (10, 22), f, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    x = 10
    for p in range(1, people + 1):
        on = labels.is_on(p)
        txt = f"[{p}] {'SPEAKING' if on else 'quiet'}"
        (tw, _), _ = cv2.getTextSize(txt, f, 0.6, 2)
        if on:
            cv2.rectangle(img, (x - 4, 32), (x + tw + 4, 54), COLORS[p], -1)
        cv2.putText(img, txt, (x, 49), f, 0.6, (0, 0, 0) if on else COLORS[p], 2, cv2.LINE_AA)
        x += tw + 24
    cv2.putText(img, "1-9 person  space pause  j/l -/+2s  ,/. frame  -/= speed  u undo  s save  q quit",
                (10, H - people * 10 - 10), f, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    # timeline: one row per person, finished segments plus the open one up to now
    dur = video.duration_s
    y0 = H - people * 10
    cv2.rectangle(img, (0, y0), (W, H), (30, 30, 30), -1)
    rows = [(s.t_start, s.t_end, s.person) for s in labels.segments]
    rows += [(t0, t, p) for p, t0 in labels.open.items() if t > t0]
    for t0, t1, p in rows:
        y = y0 + (p - 1) * 10
        cv2.rectangle(img, (int(t0 / dur * W), y + 1), (max(int(t1 / dur * W), int(t0 / dur * W) + 1), y + 9),
                      COLORS[p], -1)
    xp = int(t / dur * W)
    cv2.line(img, (xp, y0 - 4), (xp, H), (255, 255, 255), 2)
    return img


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--source", required=True, help="video file (not a camera)")
    p.add_argument("--labels", help="CSV to write (default: data/labels/<clip>.csv)")
    p.add_argument("--people", type=int, default=2, help="how many people to offer keys for (default 2)")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed (default 1)")
    p.add_argument("--start", type=float, default=0.0, help="start at this many seconds")
    p.add_argument("--width", type=int, default=1280, help="window width in pixels (default 1280)")
    p.add_argument("--fresh", action="store_true", help="ignore an existing CSV instead of continuing it")
    p.add_argument("--no-audio", action="store_true")
    args = p.parse_args(argv)
    if not 1 <= args.people <= 9:
        p.error("--people must be 1..9")

    src = parse_source(args.source)
    if isinstance(src, int):
        p.error("label.py labels video files, not cameras")
    out = Path(args.labels) if args.labels else repo_path("data/labels") / (Path(src).stem + ".csv")
    labels = Labels.load(out) if out.exists() and not args.fresh else Labels()
    if labels.segments:
        print(f"continuing {out}: {len(labels.segments)} segments")
    audio = None if args.no_audio else Audio.load(src)
    win = f"label: {Path(src).name}"

    with VideoSource(src, load_config()["camera"], SessionClock()) as video:
        frames = iter(video)
        speed, playing, ended = args.speed, True, False
        t, frame = 0.0, None
        anchor = (time.perf_counter(), 0.0)

        def goto(sec: float) -> None:
            """Jump to a time and show that frame."""
            nonlocal frames, t, frame, ended
            video.seek(int(round(sec * video.fps)))
            frames = iter(video)
            nxt = next(frames, None)
            if nxt is not None:
                t, frame = nxt
                ended = False

        def restart() -> None:
            """Re-anchor the wall clock on the current frame and restart the audio there."""
            nonlocal anchor
            anchor = (time.perf_counter(), t)
            if audio:
                audio.play_from(t, speed) if playing and not ended else audio.stop()

        goto(args.start)
        restart()
        while True:
            if playing and not ended:
                target = anchor[1] + (time.perf_counter() - anchor[0]) * speed
                while t + 1.0 / video.fps <= target:
                    nxt = next(frames, None)
                    if nxt is None:
                        ended, playing = True, False
                        break
                    t, frame = nxt
                if target - t > 0.5:   # decoding fell behind the clock: slip the clock, keep audio in sync
                    restart()
            cv2.imshow(win, draw(fit(frame, args.width), t, video, labels, args.people,
                                 playing, speed, audio is not None, ended))
            key = cv2.waitKey(1 if playing else 25) & 0xFF
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            ch = chr(key) if key < 128 else ""
            if ch == "q":
                break
            elif ch == " ":
                if ended:
                    goto(0.0)
                playing = not playing
                restart()
            elif ch.isdigit() and 1 <= int(ch) <= args.people:
                print(f"{t:8.3f}  person {ch} {labels.toggle(int(ch), t)}")
            elif ch == "j":
                goto(t - SEEK_S)
                restart()
            elif ch == "l":
                goto(t + SEEK_S)
                restart()
            elif ch == ",":
                playing = False
                goto(t - 1.0 / video.fps)
                restart()
            elif ch == ".":
                playing = False
                nxt = next(frames, None)
                if nxt is None:
                    ended = True
                else:
                    t, frame = nxt
                restart()
            elif ch == "-":
                speed = max(0.25, speed / 2)
                restart()
            elif ch == "=":
                speed = min(4.0, speed * 2)
                restart()
            elif ch == "u":
                seg = labels.undo()
                print(f"removed person {seg.person} {seg.t_start:.3f}-{seg.t_end:.3f} s" if seg else "nothing to undo")
            elif ch == "s":
                labels.save(out)
                print(f"saved {len(labels.segments)} segments -> {out}")
        closed = labels.close_all(t)
        if closed:
            print(f"closed {closed} open segment(s) at {t:.3f} s")
    if audio:
        audio.stop()
    cv2.destroyAllWindows()
    labels.save(out)
    print(f"saved {len(labels.segments)} segments -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
