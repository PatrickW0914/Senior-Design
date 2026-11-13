"""track.py - face tracking with stable IDs (Task 5) and re-identification (Task 12).

Contract
--------
FaceTracker(cfg["track"], fps).update(detections, frame, t) -> list[Track], one per face seen in
this frame, each carrying the detection's box, score and keypoints plus a track_id that stays
the same for the same person across frames. It becomes FaceObs.track_id, the identity the
voice-to-face linker accumulates evidence on, so an ID switch corrupts that evidence.
When re-ID (Task 12) decides a new track is an old person, call
VoiceFaceLinker.merge_tracks(old, new).

Tracking is ByteTrack through `supervision`: IoU matching with a Kalman motion model, so IDs
follow people through crossings, and a track lives on for track.max_lost_s without a detection
(occlusion, a missed frame) and picks its ID back up when the face returns near where the
motion model expects it. While a track is lost its velocity decays (track.lost_velocity_decay
per frame), so a head that was nodding when it got hidden is expected roughly where it was,
not a face-width away. ByteTrack only re-finds a lost track by box overlap, which fails once the
person shifted by about a face height, so on top of it a brand-new track that starts within
track.reacquire_radius_h face heights of where an ID vanished less than max_lost_s ago, while
that ID is not on screen, takes that ID over. A face that moved further, or stayed hidden
longer, gets a new ID; merging that with the old one is re-ID's job (Task 12).
A new face's track starts on its second frame, not its first.

supervision deprecates ByteTrack from 0.28 and removes it in 0.31 (the replacement is the
`trackers` package, `ByteTrackTracker`, with update() instead of update_with_detections()).
We pin supervision 0.30.9; move when the pin moves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv

from detect import Detection


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]   # (x1, y1, x2, y2) in pixels, this frame's detection
    score: float
    keypoints: np.ndarray | None = None        # (K, 2) pixel landmarks from the detection

    @property
    def cx(self) -> float:
        return 0.5 * (self.bbox[0] + self.bbox[2])

    @property
    def cy(self) -> float:
        return 0.5 * (self.bbox[1] + self.bbox[3])

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


class FaceTracker:
    def __init__(self, cfg: dict, fps: float = 30.0):
        self.cfg = cfg
        self.fps = float(fps) if fps and fps > 0 else 30.0
        max_lost_s = float(cfg.get("max_lost_s", 2.0))
        # supervision scales lost_track_buffer by frame_rate / 30, so this buffer means max_lost_s at any fps
        self._tracker = sv.ByteTrack(
            track_activation_threshold=float(cfg.get("activation_threshold", 0.5)),
            lost_track_buffer=max(1, round(30.0 * max_lost_s)),
            minimum_matching_threshold=float(cfg.get("match_threshold", 0.8)),
            frame_rate=self.fps,
            minimum_consecutive_frames=int(cfg.get("min_consecutive_frames", 1)))
        self.frames = 0
        self.lost_velocity_decay = float(cfg.get("lost_velocity_decay", 0.8))
        self.reacquire_radius_h = float(cfg.get("reacquire_radius_h", 1.5))
        self._max_lost_frames = max(1, round(max_lost_s * self.fps))
        self._id_offset = 0        # supervision restarts IDs at 1 on reset(); we keep them unique
        self._max_id = 0
        self._alias: dict[int, int] = {}          # raw ByteTrack id -> the id it took over
        self._known_raw: set[int] = set()
        self._last_seen: dict[int, tuple[int, float, float, float]] = {}   # id -> (frame, cx, cy, h)
        self._prev_ids: set[int] = set()
        self._lost: dict[int, tuple[int, float, float, float]] = {}        # id off screen -> where it was

    def update(self, detections: list[Detection], frame: np.ndarray, t: float) -> list[Track]:
        self.frames += 1
        if detections:
            dets = sv.Detections(
                xyxy=np.array([d.bbox for d in detections], dtype=float).reshape(-1, 4),
                confidence=np.array([d.score for d in detections], dtype=float),
                class_id=np.zeros(len(detections), dtype=int),
                data={"idx": np.arange(len(detections))})      # to find each track's detection again
        else:
            dets = sv.Detections.empty()
        out = self._tracker.update_with_detections(dets)
        for lost in self._tracker.lost_tracks:        # Kalman state: x, y, aspect, h, then their velocities
            if getattr(lost, "mean", None) is not None:
                lost.mean[4:6] *= self.lost_velocity_decay
        raw = []
        for k in range(len(out)):
            det = detections[int(out.data["idx"][k])]
            raw.append((int(out.tracker_id[k]) + self._id_offset, tuple(float(v) for v in out.xyxy[k]),
                        float(out.confidence[k]), det.keypoints))
        tracks = self._assign_ids(raw)
        tracks.sort(key=lambda tr: tr.track_id)
        return tracks

    def _assign_ids(self, raw: list[tuple]) -> list[Track]:
        """Turn raw ByteTrack ids into ours: apply take-overs, let new tracks adopt a nearby lost id."""
        now = self.frames
        self._lost = {i: v for i, v in self._lost.items() if now - v[0] <= self._max_lost_frames}
        old, new = [], []
        for item in raw:
            (old if item[0] in self._known_raw else new).append(item)
        ids_now = {self._alias.get(item[0], item[0]) for item in old}
        for raw_id, bbox, score, kps in new:
            self._known_raw.add(raw_id)
            cx, cy, h = 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]), bbox[3] - bbox[1]
            best = None
            for lost_id, (_, lx, ly, lh) in (self._lost.items() if self.reacquire_radius_h > 0 else ()):
                if lost_id in ids_now or not 0.5 <= h / max(lh, 1e-6) <= 2.0:
                    continue
                d = float(np.hypot(cx - lx, cy - ly))
                if d <= self.reacquire_radius_h * max(h, lh) and (best is None or d < best[0]):
                    best = (d, lost_id)
            if best is not None:
                self._alias[raw_id] = best[1]
                del self._lost[best[1]]
                ids_now.add(best[1])
        tracks, taken = [], set()
        for raw_id, bbox, score, kps in old + new:
            track_id = self._alias.get(raw_id, raw_id)
            if track_id in taken:               # the original came back by itself: the take-over was wrong
                self._alias.pop(raw_id, None)
                track_id = raw_id
            taken.add(track_id)
            self._max_id = max(self._max_id, track_id)
            self._last_seen[track_id] = (now, 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]), bbox[3] - bbox[1])
            tracks.append(Track(track_id, bbox, score, kps))
        for gone in self._prev_ids - taken:
            self._lost[gone] = self._last_seen[gone]
        for back in taken:
            self._lost.pop(back, None)
        self._prev_ids = taken
        return tracks

    def reset(self) -> None:
        """Forget every track (new clip or camera). IDs keep counting up."""
        self._tracker.reset()
        self._id_offset = self._max_id
        self.frames = 0
        self._alias.clear()
        self._known_raw.clear()
        self._last_seen.clear()
        self._prev_ids = set()
        self._lost.clear()
