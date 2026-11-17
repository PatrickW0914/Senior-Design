"""labels.py - who speaks when, as (t_start, t_end, person) segments (Task 3).

The CSV written by tools/label.py and read by the evaluation and the mock-audio generator:

    t_start,t_end,person
    1.200,4.733,1
    3.900,6.100,2

Times are seconds in the clip's time base (frame_index / fps, see capture.py). Segments of
different people may overlap; segments of one person may touch or overlap too if a clip was
labeled twice over, so use speaking_at() rather than assuming they are disjoint.

FaceLabels is the second form, for clips where people move or the shot changes (the datasets
tools/import_dataset.py brings in): data/labels/<clip>.faces.csv, one row per labeled person
per labeled frame, with the box that person is in:

    t,x1,y1,x2,y2,speaking,person,box
    0.040,412.0,96.5,530.2,251.0,1,3,face

box is "face" (a face box) or "body" (an upper-body box that contains the face); a tracked
face gets the truth of the labeled box it falls in.
"""

from __future__ import annotations

import bisect
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Segment:
    t_start: float
    t_end: float
    person: int


class Labels:
    """Finished segments plus, while labeling, one open segment per person."""

    def __init__(self, segments: list[Segment] | None = None):
        self.segments: list[Segment] = list(segments or [])   # in the order they were made
        self.open: dict[int, float] = {}                       # person -> t_start of a segment in progress

    # --- labeling -------------------------------------------------------------------------
    def is_on(self, person: int) -> bool:
        return person in self.open

    def toggle(self, person: int, t: float) -> str:
        """Start a segment for `person` at t, or finish the open one. Returns what happened."""
        if person not in self.open:
            self.open[person] = t
            return "started"
        t0 = self.open.pop(person)
        if t <= t0:   # stopped at or before the start (after seeking back): nothing to keep
            return f"discarded (stopped at {t:.3f} s, before its start at {t0:.3f} s)"
        self.segments.append(Segment(t0, t, person))
        return f"stopped ({t - t0:.2f} s)"

    def close_all(self, t: float) -> int:
        """Finish every open segment at t (end of clip, or quitting). Returns how many."""
        n = 0
        for person in list(self.open):
            if self.toggle(person, t).startswith("stopped"):
                n += 1
        return n

    def undo(self) -> Segment | None:
        """Remove the most recently finished segment."""
        return self.segments.pop() if self.segments else None

    # --- queries --------------------------------------------------------------------------
    def speaking_at(self, t: float) -> set[int]:
        return {s.person for s in self.segments if s.t_start <= t < s.t_end}

    def people(self) -> set[int]:
        return {s.person for s in self.segments}

    def sorted(self) -> list[Segment]:
        return sorted(self.segments, key=lambda s: (s.t_start, s.t_end, s.person))

    # --- files ----------------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> Labels:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("t_start")]
        return cls([Segment(float(r["t_start"]), float(r["t_end"]), int(r["person"])) for r in rows])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t_start", "t_end", "person"])
            for s in self.sorted():
                w.writerow([f"{s.t_start:.3f}", f"{s.t_end:.3f}", s.person])


@dataclass(frozen=True)
class FaceLabel:
    t: float
    x1: float
    y1: float
    x2: float
    y2: float
    speaking: bool
    person: int
    box: str = "face"            # face | body

    def anchor(self) -> tuple[float, float]:
        """Where the face centre is expected inside this box."""
        cx = 0.5 * (self.x1 + self.x2)
        return (cx, 0.5 * (self.y1 + self.y2)) if self.box == "face" else (cx, self.y1 + 0.25 * (self.y2 - self.y1))

    def contains(self, x: float, y: float, pad: float = 0.15) -> bool:
        px, py = pad * (self.x2 - self.x1), pad * (self.y2 - self.y1)
        return self.x1 - px <= x <= self.x2 + px and self.y1 - py <= y <= self.y2 + py


class FaceLabels:
    """Per-frame boxes with a speaking flag; see the module docstring for the file format."""

    def __init__(self, rows: list[FaceLabel] | None = None):
        self.rows: list[FaceLabel] = list(rows or [])
        self._by_t: dict[float, list[FaceLabel]] = {}
        for r in self.rows:
            self._by_t.setdefault(r.t, []).append(r)
        self._times = sorted(self._by_t)

    def at(self, t: float, tol: float = 0.06) -> list[FaceLabel]:
        """The labels of the labeled frame nearest to t, or [] when none is within tol seconds."""
        if not self._times:
            return []
        i = bisect.bisect_left(self._times, t)
        near = [self._times[j] for j in (i - 1, i) if 0 <= j < len(self._times)]
        best = min(near, key=lambda u: abs(u - t))
        return self._by_t[best] if abs(best - t) <= tol else []

    def speaking_for(self, t: float, x: float, y: float, tol: float = 0.06) -> bool | None:
        """Is the labeled person whose box holds the point (x, y) speaking at t? None = no such label."""
        inside = [r for r in self.at(t, tol) if r.contains(x, y)]
        if not inside:
            return None
        def dist(r: FaceLabel) -> float:
            ax, ay = r.anchor()
            return ((x - ax) ** 2 + (y - ay) ** 2) ** 0.5 / max(r.x2 - r.x1, 1e-6)
        return min(inside, key=dist).speaking

    def people(self) -> set[int]:
        return {r.person for r in self.rows}

    def span(self) -> tuple[float, float]:
        return (self._times[0], self._times[-1]) if self._times else (0.0, 0.0)

    @classmethod
    def load(cls, path: str | Path) -> FaceLabels:
        with open(path, newline="", encoding="utf-8") as fh:
            return cls([FaceLabel(float(r["t"]), float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]),
                                  r["speaking"] == "1", int(r["person"]), r.get("box") or "face")
                        for r in csv.DictReader(fh) if r.get("t")])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "x1", "y1", "x2", "y2", "speaking", "person", "box"])
            for r in sorted(self.rows, key=lambda r: (r.t, r.person)):
                w.writerow([f"{r.t:.3f}", f"{r.x1:.1f}", f"{r.y1:.1f}", f"{r.x2:.1f}", f"{r.y2:.1f}",
                            int(r.speaking), r.person, r.box])
