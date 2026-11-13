"""labels.py - who speaks when, as (t_start, t_end, person) segments (Task 3).

The CSV written by tools/label.py and read by the evaluation and the mock-audio generator:

    t_start,t_end,person
    1.200,4.733,1
    3.900,6.100,2

Times are seconds in the clip's time base (frame_index / fps, see capture.py). Segments of
different people may overlap; segments of one person may touch or overlap too if a clip was
labeled twice over, so use speaking_at() rather than assuming they are disjoint.
"""

from __future__ import annotations

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
