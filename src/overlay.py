"""overlay.py - debug visualization (Task 7).

Contract
--------
Overlay(cfg["overlay"], angles).draw(frame, t, faces, mouth=, speaking=, names=, doa=, note=)
returns the annotated picture: the frame scaled down to overlay.width (or up to overlay.min_width,
so the labels fit on a small source) with a strip added underneath. Its size depends only on the input size, so it can go to a window and to a
video file alike. What it draws:

  box per face   green while that track is speaking, grey otherwise (or while speaking is unknown)
  label          "#id name  azimuth", and the speaking probability under the box
  live graph     the last overlay.graph_seconds of the face's mouth-openness value (Task 8); the
                 speaking probability when there is no mouth value; "mouth n/a" when neither
  OVERLAP        a red banner (bottom centre) and a red frame while two or more faces are speaking
  bottom strip   reserved for the mic array's sound-direction spectrum. It is laid out in camera
                 azimuth (ticks where each angle falls in the image, a marker under each face), so
                 spectrum peaks will sit under the faces. Pass doa=(grid_deg, power) to plot one.

Arguments: faces is the list of FaceObs for this frame; mouth maps track_id -> mouth-openness;
speaking is the set of track ids speaking now, or None when not known (no speaking stage yet);
names maps track_id -> name (overlay.names in the config is the default).

draw(frame, t, faces) is the one-call form: one shared Overlay, speaking = asd_prob >= 0.5.
VideoSaver writes annotated frames to an .mp4.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv_env  # noqa: F401  (must stay above cv2)
import cv2
import numpy as np

from av_event_log import FaceObs

GREEN, GREY, RED = (60, 200, 60), (190, 190, 190), (40, 40, 220)        # BGR
YELLOW, CYAN, WHITE, DARK, TICK = (0, 220, 255), (255, 200, 0), (255, 255, 255), (25, 25, 25), (150, 150, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Overlay:
    def __init__(self, cfg: dict | None = None, angles=None):
        cfg = cfg or {}
        self.max_width = int(cfg.get("width", 1280))
        self.min_width = int(cfg.get("min_width", 640))
        self.graph_s = float(cfg.get("graph_seconds", 3.0))
        self.strip_h = int(cfg.get("strip_height", 90))
        self.names = {int(k): str(v) for k, v in (cfg.get("names") or {}).items()}
        self.angles = angles                       # geometry.AzimuthModel for the source frame size, or None
        self._hist: dict[int, deque] = {}          # track id -> (t, value) samples for the graph
        self._peak: dict[int, float] = {}
        self._last_t = None

    # --- sizes ----------------------------------------------------------------------------
    def scale_for(self, frame_w: int) -> float:
        """Shrink a wide frame to max_width; enlarge a small one to min_width so the labels fit."""
        if frame_w < self.min_width:
            return self.min_width / frame_w
        return min(1.0, self.max_width / frame_w)

    def output_size(self, frame_w: int, frame_h: int) -> tuple[int, int]:
        """(width, height) of what draw() returns for this frame size."""
        s = self.scale_for(frame_w)
        return round(frame_w * s), round(frame_h * s) + self.strip_h

    # --- drawing helpers ------------------------------------------------------------------
    @staticmethod
    def _shade(img, x1, y1, x2, y2, alpha=0.6, color=(0, 0, 0)):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            roi = img[y1:y2, x1:x2]
            img[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, color), alpha, 0)

    @staticmethod
    def _text(img, text, x, y, scale, color, thick=1, bg=None, pad=3):
        """Text with its baseline at (x, y); optional filled background. Returns (width, height)."""
        (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
        if bg is not None:
            cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + base + pad - 1), bg, -1)
        cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)
        return tw, th + base

    def _remember(self, t: float, values: dict[int, float]) -> None:
        if self._last_t is not None and t < self._last_t:      # new clip or a seek backwards
            self._hist.clear()
            self._peak.clear()
        self._last_t = t
        for tid, v in values.items():
            self._hist.setdefault(tid, deque()).append((t, float(v)))
            self._peak[tid] = max(self._peak.get(tid, 0.0), float(v))
        for tid in list(self._hist):
            h = self._hist[tid]
            while h and h[0][0] < t - self.graph_s:
                h.popleft()
            if not h:
                del self._hist[tid]
                self._peak.pop(tid, None)

    def _graph(self, img, tid, t, x, y, w, h, k, label, top=None):
        """A caption row ("mouth 0.04") over a plot of the last graph_seconds."""
        cap = int(15 * k)                                        # caption row, so text never covers the line
        self._shade(img, x, y, x + w, y + h, 0.7)
        cv2.rectangle(img, (x, y), (x + w, y + h), TICK, 1)
        hist = self._hist.get(tid)
        if not hist:
            self._text(img, f"{label} n/a", x + 4, y + int(12 * k), 0.42 * k, TICK)
            return
        self._text(img, f"{label} {hist[-1][1]:.2f}", x + 4, y + int(12 * k), 0.42 * k, WHITE)
        top = top or max(self._peak.get(tid, 0.0), 1e-6)
        y_lo, y_hi = y + h - 3, y + cap + 1
        pts = [(int(x + 1 + (ti - (t - self.graph_s)) / self.graph_s * (w - 3)),
                int(y_lo - min(1.0, max(0.0, v / top)) * (y_lo - y_hi))) for ti, v in hist]
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, YELLOW, max(1, round(1.5 * k)), cv2.LINE_AA)
        else:
            cv2.circle(img, pts[0], 2, YELLOW, -1)

    # --- the picture ----------------------------------------------------------------------
    def draw(self, frame: np.ndarray, t: float, faces: list[FaceObs], mouth: dict | None = None,
             speaking: set | None = None, names: dict | None = None, doa=None, note: str = "") -> np.ndarray:
        fh, fw = frame.shape[:2]
        s = self.scale_for(fw)
        w, h = round(fw * s), round(fh * s)
        k = max(0.6, w / 1280)                                   # size of text and lines
        img = np.zeros((h + self.strip_h, w, 3), np.uint8)
        img[:h] = frame if s == 1 else cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        names = {**self.names, **(names or {})}
        present = {f.track_id for f in faces}
        talking = None if speaking is None else set(speaking) & present

        # graph series: mouth value when there is one, else the speaking probability when known
        mouth = mouth or {}
        series = {f.track_id: mouth[f.track_id] for f in faces if f.track_id in mouth}
        series_label = "mouth"
        if not series and speaking is not None:
            series, series_label = {f.track_id: f.asd_prob for f in faces}, "p"
        self._remember(t, series)

        header_h = int(30 * k)
        for f in faces:
            if f.bbox is None:
                continue
            x1, y1, x2, y2 = (int(round(v * s)) for v in f.bbox)
            on = talking is not None and f.track_id in talking
            color = GREEN if on else GREY
            cv2.rectangle(img, (x1, y1), (x2, y2), color, max(2, round((4 if on else 2) * k)))
            label = f"#{f.track_id}" + (f" {names[f.track_id]}" if f.track_id in names else "") + f"  {f.azimuth_deg:+.1f} deg"
            (tw, th), base = cv2.getTextSize(label, FONT, 0.6 * k, max(1, round(2 * k)))
            ly = y1 - int(6 * k) if y1 - th - int(10 * k) > header_h else y1 + th + int(6 * k)
            self._text(img, label, x1, ly, 0.6 * k, (0, 0, 0), max(1, round(2 * k)), bg=color)
            below = y2 + int(18 * k)
            if talking is not None:
                self._text(img, f"p {f.asd_prob:.2f}" + ("  SPEAKING" if on else ""), x1, below, 0.5 * k,
                           color, max(1, round(1.5 * k)), bg=(0, 0, 0))
                below += int(8 * k)
            gw, gh = max(int(110 * k), x2 - x1), int(50 * k)
            gy = below if below + gh <= h else max(header_h, y1 - th - int(14 * k) - gh)
            gx = min(max(0, x1), w - gw)
            self._graph(img, f.track_id, t, gx, gy, gw, gh, k, series_label, top=1.0 if series_label == "p" else None)

        # header
        self._shade(img, 0, 0, w, header_h, 0.7)
        who = "n/a" if talking is None else (", ".join(f"#{i}" for i in sorted(talking)) or "nobody")
        self._text(img, f"t={t:.2f}s   faces {len(faces)}   speaking: {who}" + (f"   {note}" if note else ""),
                   int(8 * k), int(21 * k), 0.55 * k, WHITE, max(1, round(1.5 * k)))
        if talking is not None and len(talking) >= 2:
            # bottom centre, where faces rarely are (top centre covered face labels), plus a red frame
            bw, bh = int(260 * k), int(46 * k)
            bx, by = (w - bw) // 2, h - bh - int(10 * k)
            cv2.rectangle(img, (0, header_h), (w - 1, h - 1), RED, max(3, round(5 * k)))
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), RED, -1)
            (tw, th), _ = cv2.getTextSize("OVERLAP", FONT, 1.1 * k, max(2, round(3 * k)))
            cv2.putText(img, "OVERLAP", (bx + (bw - tw) // 2, by + (bh + th) // 2), FONT, 1.1 * k, WHITE,
                        max(2, round(3 * k)), cv2.LINE_AA)

        self._strip(img, h, w, s, k, faces, talking, doa)
        return img

    def _strip(self, img, y0, w, s, k, faces, talking, doa) -> None:
        """Bottom strip: azimuth axis aligned with the image columns, face markers, DOA spectrum or placeholder."""
        img[y0:] = DARK
        cv2.line(img, (0, y0), (w, y0), TICK, 1)
        base = y0 + self.strip_h - int(20 * min(k, 1.0)) - 2
        cv2.line(img, (0, base), (w, base), TICK, 1)
        if self.angles is not None:
            for az in range(-90, 91, 10):
                x = self.angles.x_at_azimuth(az) * s
                if 0 <= x < w:
                    cv2.line(img, (int(x), base - 5), (int(x), base + 5), TICK, 1)
                    self._text(img, f"{az:+d}" if az else "0", int(x) - int(10 * k), base + int(17 * min(k, 1.0)), 0.4 * k, TICK)
        for f in faces:                                         # where each face is, on the same axis
            if f.bbox is None:
                continue
            x = int(0.5 * (f.bbox[0] + f.bbox[2]) * s)
            color = GREEN if talking is not None and f.track_id in talking else GREY
            tri = np.array([[x - int(7 * k), y0 + 3], [x + int(7 * k), y0 + 3], [x, y0 + 3 + int(11 * k)]], np.int32)
            cv2.fillPoly(img, [tri], color)
            self._text(img, f"#{f.track_id}", x + int(10 * k), y0 + int(15 * k), 0.42 * k, color)
        top = y0 + int(20 * k)
        if doa is not None and self.angles is not None:
            grid, power = np.asarray(doa[0], float), np.asarray(doa[1], float)
            peak = float(power.max()) if power.size and power.max() > 0 else 1.0
            pts = [(int(self.angles.x_at_azimuth(a) * s), int(base - 2 - (p / peak) * (base - 2 - top)))
                   for a, p in zip(grid, power) if abs(a) < 89 and 0 <= self.angles.x_at_azimuth(a) * s < w]
            if len(pts) > 1:
                cv2.polylines(img, [np.array(pts, np.int32)], False, CYAN, max(1, round(2 * k)), cv2.LINE_AA)
            self._text(img, "sound direction (mic array)", int(8 * k), base - int(8 * k), 0.45 * k, CYAN)
        else:
            self._text(img, "sound direction (mic array): not connected yet", int(8 * k), base - int(8 * k), 0.45 * k, TICK)


_shared = Overlay()


def draw(frame: np.ndarray, t: float, faces: list[FaceObs]) -> np.ndarray:
    """One-call form: boxes, ids, azimuths and speaking probability, speaking = asd_prob >= 0.5."""
    return _shared.draw(frame, t, faces, speaking={f.track_id for f in faces if f.asd_prob >= 0.5})


class VideoSaver:
    """Writes frames to an .mp4 (mp4v). Opens on the first frame, which fixes the size."""

    def __init__(self, path: str | Path, fps: float):
        self.path = Path(path)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self._writer = None
        self._size = None
        self.frames = 0

    def write(self, img: np.ndarray) -> None:
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._size = (img.shape[1], img.shape[0])
            self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self._size)
            if not self._writer.isOpened():
                raise RuntimeError(f"could not open {self.path} for writing")
        if (img.shape[1], img.shape[0]) != self._size:
            img = cv2.resize(img, self._size)
        self._writer.write(img)
        self.frames += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> VideoSaver:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
