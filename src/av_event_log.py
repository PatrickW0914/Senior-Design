#!/usr/bin/env python3
"""
av_event_log.py - Timestamped audio-visual event log with voice-to-face linking.

Where it sits in the system
---------------------------
  camera -> face tracker (ByteTrack) -> active-speaker model (TalkNet / Light-ASD) --+
  mic array -> VAD (Silero) + DOA spectrum (SRP-PHAT / GCC-PHAT) --------------------+--> SpeakingFusion
                                                                                     |         |
  diarizer (diart / pyannote) -> VoiceSegment ---------> VoiceFaceLinker <-----------+         v
                                                              |                         SegmentBuilder
  ASR (faster-whisper / WhisperX) -> Word ---------+          |                                |
                                                   v          v                                v
                                                EventLog  (live JSONL + final speaker-attributed transcript)

Key ideas
---------
* Every module stamps its output with ONE shared session clock (SessionClock).
* SpeakingFusion turns per-face ASD probability + audio direction + VAD into a smoothed,
  hysteresis-gated "speaking" flag per face track, and flags off-camera voices.
* VoiceFaceLinker keeps a short history of who was visibly speaking. When the diarizer
  reports "VOICE_x spoke from t1 to t2" (usually seconds late), it credits VOICE_x to the
  faces that were speaking in that window, then solves a one-to-one assignment
  (Hungarian algorithm) between voices and faces. Overlapping-speech frames split their
  credit, so clean solo turns dominate the evidence.
* Once a voice is linked to a face, the voice identifies that person even when their face
  is lost (turned to the slides), and transcript words carrying a voice label are
  attributed to the right person.

Outputs (prefix configurable)
-----------------------------
  <prefix>_events.jsonl      live log: session_start, segment, link, session_end
  <prefix>_transcript.json   final segments with speaker, direction, overlap flag, text
  <prefix>_transcript.txt    human-readable transcript, ready to feed to an LLM summarizer

Run  `python av_event_log.py`  for a synthetic end-to-end demo (needs numpy + scipy).

Live wiring (sketch)
--------------------
    clock = SessionClock(camera_latency_s=0.09)          # from a clap test
    pipe  = AVSpeakerPipeline(doa_grid_deg=np.arange(-90, 91, 2), out_prefix="talk1",
                              names={1: "Alice", 2: "Bob"})   # optional enrollment

    # sounddevice callback thread (keep it light: compute VAD/DOA in a worker if needed)
    def audio_cb(indata, frames, t_info, status):
        t = clock.audio_ts(t_info.inputBufferAdcTime)
        pipe.on_audio_frame(AudioFrame(t, vad(indata), srp_phat(indata), overlap_prob(indata)))

    # video loop: process each frame ~0.2-0.3 s late so audio around it has arrived
    ok = cap.grab(); t = clock.video_ts(time.perf_counter()); _, img = cap.retrieve()
    faces = [FaceObs(trk.id, pixel_to_azimuth(trk.cx), asd_prob[trk.id]) for trk in tracks]
    pipe.on_video_frame(t, faces)

    # diarizer (e.g. diart) callback, in session time
    pipe.on_voice_segment(VoiceSegment(t_start, t_end, "VOICE_3", confidence))

    # ASR words (faster-whisper word_timestamps=True, or WhisperX with speaker labels)
    pipe.on_words([Word(w.start, w.end, w.word, voice_id=w_speaker) for w in words])

    segments, unassigned = pipe.close()
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

OFF_CAMERA = -1  # pseudo track id for a voice with no visible face at its direction


# ---------------------------------------------------------------------------
# Shared clock and helpers
# ---------------------------------------------------------------------------
class SessionClock:
    """One monotonic time base (seconds since session start) for every module."""

    def __init__(self, camera_latency_s: float = 0.0):
        self._t0 = time.perf_counter()
        self.camera_latency_s = camera_latency_s  # measure with a clap / LED+beep test
        self._audio_offset: float | None = None   # perf_counter - PortAudio stream.time

    def now(self) -> float:
        return time.perf_counter() - self._t0

    def video_ts(self, grab_perf_counter: float) -> float:
        """Pass time.perf_counter() taken right after cap.grab()."""
        return grab_perf_counter - self._t0 - self.camera_latency_s

    def sync_audio_stream(self, stream_time_now: float) -> None:
        """Call once after opening the sounddevice stream: clock.sync_audio_stream(stream.time)."""
        self._audio_offset = time.perf_counter() - stream_time_now

    def audio_ts(self, input_buffer_adc_time: float) -> float:
        """Convert PortAudio's callback time.inputBufferAdcTime to session time.

        Some host APIs report 0 for inputBufferAdcTime; in that case fall back to
        clock.now() - frames / samplerate inside your callback.
        """
        if self._audio_offset is None:
            raise RuntimeError("call sync_audio_stream(stream.time) first")
        return input_buffer_adc_time + self._audio_offset - self._t0


def angdiff(a, b):
    """Signed angular difference a - b in degrees, wrapped to [-180, 180)."""
    return (np.asarray(a, dtype=float) - b + 180.0) % 360.0 - 180.0


def fmt_ts(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _subtract(intervals, cuts):
    """Remove the `cuts` intervals from `intervals`; returns what is left."""
    out = list(intervals)
    for c0, c1 in sorted(cuts):
        nxt = []
        for a, b in out:
            if c1 <= a or c0 >= b:
                nxt.append((a, b))
                continue
            if c0 > a:
                nxt.append((a, c0))
            if c1 < b:
                nxt.append((c1, b))
        out = nxt
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------
@dataclass
class FaceObs:
    track_id: int
    azimuth_deg: float           # from pixel_to_azimuth(face centre x)
    asd_prob: float              # active-speaker probability 0..1
    bbox: tuple | None = None    # (x1, y1, x2, y2), optional


@dataclass
class AudioFrame:
    t: float                     # session time at frame centre
    vad_prob: float              # 0..1
    doa_power: np.ndarray        # spatial spectrum P(theta) on AudioBuffer.grid_deg
    overlap_prob: float = 0.0    # optional overlapped-speech probability


@dataclass
class VoiceSegment:
    t_start: float
    t_end: float
    voice_id: str                # anonymous diarization cluster, e.g. "VOICE_3"
    confidence: float = 1.0


@dataclass
class Word:
    t_start: float
    t_end: float
    text: str
    voice_id: str | None = None  # speaker label from WhisperX / diarization, if any
    track_id: int | None = None  # set when the word came from a beam steered at a face


@dataclass
class FrameResult:
    t: float
    scores: dict
    speaking: dict
    azimuths: dict
    vad: float
    overlap: bool
    offcamera: bool
    offcamera_azimuth: float | None = None

    @property
    def active_tracks(self):
        return [k for k, on in self.speaking.items() if on]


@dataclass
class Segment:
    seg_id: int
    track_id: int                # OFF_CAMERA (-1) for unlinked off-camera voices
    t_start: float
    t_end: float
    azimuth_deg: float | None
    confidence: float
    overlap: bool
    source: str = "av"           # av | doa | voice | doa+voice (combinations after merging)
    speaker: str = ""
    voice_id: str | None = None
    text: str = ""

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


# ---------------------------------------------------------------------------
# Audio ring buffer
# ---------------------------------------------------------------------------
class AudioBuffer:
    """Time-indexed ring buffer of per-hop audio features (thread-safe)."""

    def __init__(self, grid_deg, horizon_s: float = 10.0):
        self.grid_deg = np.asarray(grid_deg, dtype=float)
        self.horizon_s = horizon_s
        self._frames: deque[AudioFrame] = deque()
        self._lock = threading.Lock()

    def push(self, frame: AudioFrame) -> None:
        with self._lock:
            self._frames.append(frame)
            while self._frames and frame.t - self._frames[0].t > self.horizon_s:
                self._frames.popleft()

    def window(self, t: float, half_width: float) -> list[AudioFrame]:
        with self._lock:
            return [f for f in self._frames if abs(f.t - t) <= half_width]


# ---------------------------------------------------------------------------
# Per-frame audio-visual fusion
# ---------------------------------------------------------------------------
class SpeakingFusion:
    """score = w_asd*ASD + w_doa*P(theta_face)*VAD + w_vad*VAD, smoothed with hysteresis."""

    def __init__(self, audio: AudioBuffer, w_asd=0.5, w_doa=0.4, w_vad=0.1,
                 on_threshold=0.6, off_threshold=0.4, tau_s=0.25, audio_window_s=0.2,
                 doa_tolerance_deg=8.0, peak_threshold=0.6, offcam_margin_deg=15.0,
                 forget_after_s=2.0):
        self.audio = audio
        self.w_asd, self.w_doa, self.w_vad = w_asd, w_doa, w_vad
        self.on_threshold, self.off_threshold = on_threshold, off_threshold
        self.tau_s = tau_s
        self.audio_window_s = audio_window_s
        self.doa_tolerance_deg = doa_tolerance_deg
        self.peak_threshold = peak_threshold
        self.offcam_margin_deg = offcam_margin_deg
        self.forget_after_s = forget_after_s
        self._state: dict[int, tuple[float, bool, float]] = {}  # track -> (score, on, last_t)
        self._last_t: float | None = None

    def _audio_profile(self, t):
        frames = self.audio.window(t, self.audio_window_s)
        if not frames:
            return None, 0.0, 0.0
        P = np.mean([f.doa_power for f in frames], axis=0).astype(float)
        P -= P.min()
        P /= P.max() + 1e-9
        vad = float(np.mean([f.vad_prob for f in frames]))
        ov = float(np.mean([f.overlap_prob for f in frames]))
        return P, vad, ov

    def _doa_at(self, P, az):
        near = np.abs(angdiff(self.audio.grid_deg, az)) <= self.doa_tolerance_deg
        return float(P[near].max()) if near.any() else 0.0

    def _peaks(self, P):
        left = np.r_[-np.inf, P[:-1]]
        right = np.r_[P[1:], -np.inf]
        idx = np.where((P >= left) & (P > right) & (P >= self.peak_threshold))[0]
        return self.audio.grid_deg[idx]

    def update(self, t: float, faces: list[FaceObs]) -> FrameResult:
        P, vad, ov = self._audio_profile(t)
        dt = 0.04 if self._last_t is None else max(1e-3, t - self._last_t)
        self._last_t = t
        alpha = 1.0 - math.exp(-dt / self.tau_s)

        scores, speaking, azimuths = {}, {}, {}
        for f in faces:
            doa = self._doa_at(P, f.azimuth_deg) if P is not None else 0.0
            raw = self.w_asd * f.asd_prob + self.w_doa * doa * vad + self.w_vad * vad
            s, on, _ = self._state.get(f.track_id, (0.0, False, t))
            s += alpha * (raw - s)
            on = (s > self.off_threshold) if on else (s >= self.on_threshold)
            self._state[f.track_id] = (s, on, t)
            scores[f.track_id], speaking[f.track_id] = s, on
            azimuths[f.track_id] = f.azimuth_deg

        for k in [k for k, (_, _, lt) in self._state.items() if t - lt > self.forget_after_s]:
            del self._state[k]

        offcam, off_az = False, None
        if P is not None and vad > 0.5:
            for p in self._peaks(P):
                if all(abs(float(angdiff(p, a))) > self.offcam_margin_deg for a in azimuths.values()):
                    offcam, off_az = True, float(p)
                    break

        overlap = sum(speaking.values()) >= 2 or (ov > 0.5 and vad > 0.5)
        return FrameResult(t, scores, speaking, azimuths, vad, overlap, offcam, off_az)


# ---------------------------------------------------------------------------
# Frame flags -> speaking segments
# ---------------------------------------------------------------------------
class SegmentBuilder:
    def __init__(self, gap_merge_s: float = 0.5, min_duration_s: float = 0.4):
        self.gap_merge_s = gap_merge_s
        self.min_duration_s = min_duration_s
        self._open: dict[int, dict] = {}
        self._next_id = 1

    def update(self, fr: FrameResult) -> list[Segment]:
        active = {k: (fr.azimuths[k], fr.scores[k]) for k in fr.active_tracks}
        if fr.offcamera:
            active[OFF_CAMERA] = (fr.offcamera_azimuth, fr.vad)
        for k, (az, sc) in active.items():
            o = self._open.setdefault(k, {"t_start": fr.t, "last": fr.t, "az": [], "sc": [], "ov": 0, "n": 0})
            o["last"] = fr.t
            o["az"].append(az)
            o["sc"].append(sc)
            o["ov"] += int(fr.overlap)
            o["n"] += 1
        closed = []
        for k in list(self._open):
            if k not in active and fr.t - self._open[k]["last"] > self.gap_merge_s:
                seg = self._close(k)
                if seg:
                    closed.append(seg)
        return closed

    def flush(self) -> list[Segment]:
        return [s for s in (self._close(k) for k in list(self._open)) if s]

    def _close(self, k) -> Segment | None:
        o = self._open.pop(k)
        if o["last"] - o["t_start"] < self.min_duration_s:
            return None
        seg = Segment(self._next_id, k, o["t_start"], o["last"], float(np.mean(o["az"])),
                      float(np.mean(o["sc"])), o["ov"] / o["n"] > 0.3,
                      source="doa" if k == OFF_CAMERA else "av")
        self._next_id += 1
        return seg


# ---------------------------------------------------------------------------
# Voice (diarization cluster) <-> face track linking
# ---------------------------------------------------------------------------
class VoiceFaceLinker:
    """Accumulates co-occurrence evidence and solves a one-to-one voice->face assignment."""

    def __init__(self, history_s=120.0, min_evidence_s=4.0, min_share=0.5, min_margin=0.15):
        self.history_s = history_s
        self.min_evidence_s = min_evidence_s   # seconds of a voice before we try to link it
        self.min_share = min_share             # fraction of that voice's time on the linked face
        self.min_margin = min_margin           # lead over the second-best face
        self._history: deque[tuple[float, float, dict]] = deque()
        self._pending: list[VoiceSegment] = []
        self._latest_t = -math.inf
        self.C = defaultdict(lambda: defaultdict(float))  # voice -> track -> credited seconds
        self.voice_total = defaultdict(float)             # voice -> seconds observed
        self.links: dict[str, tuple[int, float]] = {}     # voice -> (track, share)

    def add_frame(self, fr: FrameResult, dt: float) -> None:
        on = {k: fr.scores[k] for k in fr.active_tracks}
        total = sum(on.values())
        weights = {k: s / total for k, s in on.items()} if total > 0 else {}
        self._history.append((fr.t, dt, weights))
        self._latest_t = fr.t
        while self._history and fr.t - self._history[0][0] > self.history_s:
            self._history.popleft()

    def add_voice_segment(self, vs: VoiceSegment) -> None:
        self._pending.append(vs)

    def process_pending(self, force: bool = False) -> list[dict]:
        """Credit voice segments whose time span is fully covered by video frames."""
        ready = [v for v in self._pending if force or v.t_end <= self._latest_t]
        if not ready:
            return []
        self._pending = [v for v in self._pending if v not in ready]
        for vs in ready:
            for t, dt, w in self._history:
                if vs.t_start <= t <= vs.t_end:
                    self.voice_total[vs.voice_id] += dt * vs.confidence
                    for k, wk in w.items():
                        self.C[vs.voice_id][k] += dt * wk * vs.confidence
        return self._relink()

    def _relink(self) -> list[dict]:
        voices = [v for v, tot in self.voice_total.items() if tot >= self.min_evidence_s]
        tracks = sorted({k for v in voices for k in self.C[v]})
        new: dict[str, tuple[int, float]] = {}
        if voices and tracks:
            S = np.array([[self.C[v][k] / self.voice_total[v] for k in tracks] for v in voices])
            rows, cols = linear_sum_assignment(-S)
            for r, c in zip(rows, cols):
                share = float(S[r, c])
                others = np.delete(S[r], c)
                margin = share - (float(others.max()) if others.size else 0.0)
                if share >= self.min_share and margin >= self.min_margin:
                    new[voices[r]] = (tracks[c], share)
        changes = []
        for v in sorted(set(self.links) | set(new)):
            old_k = self.links.get(v, (None, 0.0))[0]
            new_k, share = new.get(v, (None, 0.0))
            if old_k != new_k:
                changes.append({"voice_id": v, "old_track": old_k, "new_track": new_k,
                                "share": round(share, 3)})
        self.links = new
        return changes

    def track_for_voice(self, voice_id: str | None) -> int | None:
        return self.links.get(voice_id, (None, 0.0))[0] if voice_id else None

    def voice_for_track(self, track_id: int) -> str | None:
        for v, (k, _) in self.links.items():
            if k == track_id:
                return v
        return None

    def merge_tracks(self, old_track: int, new_track: int) -> list[dict]:
        """Call when face re-ID says a new track is the same person as an old one."""
        for v in list(self.C):
            if old_track in self.C[v]:
                self.C[v][new_track] += self.C[v].pop(old_track)
        return self._relink()


# ---------------------------------------------------------------------------
# Event log and final transcript
# ---------------------------------------------------------------------------
class EventLog:
    def __init__(self, jsonl_path: str, names: dict[int, str] | None = None):
        self._fh = open(jsonl_path, "w", encoding="utf-8")
        self._lock = threading.Lock()
        self.names = dict(names or {})
        self._auto: dict[int, str] = {}
        self.segments: list[Segment] = []
        self.voice_segments: list[VoiceSegment] = []
        self.words: list[Word] = []

    def label(self, track_id: int) -> str:
        if track_id == OFF_CAMERA:
            return "Off-camera"
        if track_id in self.names:
            return self.names[track_id]
        if track_id not in self._auto:
            self._auto[track_id] = f"Speaker {chr(ord('A') + len(self._auto))}"
        return self._auto[track_id]

    def write(self, event: str, t: float, **data) -> None:
        rec = {"event": event, "t": round(float(t), 3), **data}
        with self._lock:
            self._fh.write(json.dumps(rec, default=_json_default) + "\n")
            self._fh.flush()

    @staticmethod
    def seg_dict(s: Segment) -> dict:
        d = asdict(s)
        for key in ("t_start", "t_end", "confidence"):
            d[key] = round(d[key], 3)
        if d["azimuth_deg"] is not None:
            d["azimuth_deg"] = round(d["azimuth_deg"], 1)
        return d

    def add_segment(self, seg: Segment, linker: VoiceFaceLinker) -> None:
        seg.speaker = self.label(seg.track_id)
        seg.voice_id = None if seg.track_id == OFF_CAMERA else linker.voice_for_track(seg.track_id)
        self.segments.append(seg)
        self.write("segment", seg.t_end, **self.seg_dict(seg))

    def log_link_changes(self, changes: list[dict], t: float) -> None:
        for c in changes:
            k = c["new_track"]
            self.write("link", t, **c, speaker=None if k is None else self.label(k))

    # -- finalization --------------------------------------------------------
    def finalize(self, linker: VoiceFaceLinker, prefix: str, gap_merge_s: float = 0.5):
        segs = sorted(self.segments, key=lambda s: s.t_start)
        segs = self._resolve_with_voices(segs, linker)
        segs = self._merge_adjacent(segs, gap_merge_s)
        for i, s in enumerate(segs, 1):
            s.seg_id = i
            s.speaker = self.label(s.track_id)
            s.voice_id = None if s.track_id == OFF_CAMERA else linker.voice_for_track(s.track_id)
        unassigned = self._attach_words(segs, linker)

        payload = {
            "speakers": {self.label(k): {"track_id": k, "voice_id": linker.voice_for_track(k)}
                         for k in sorted({s.track_id for s in segs})},
            "segments": [self.seg_dict(s) for s in segs],
            "unassigned_words": [asdict(w) for w in unassigned],
        }
        with open(f"{prefix}_transcript.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=_json_default)
        with open(f"{prefix}_transcript.txt", "w", encoding="utf-8") as fh:
            fh.write(self.transcript_text(segs, unassigned))

        end_t = max((s.t_end for s in segs), default=0.0)
        self.write("session_end", end_t, n_segments=len(segs), n_unassigned_words=len(unassigned))
        self._fh.close()
        return segs, unassigned

    def _resolve_with_voices(self, segs, linker):
        """Use linked voices to cover time when the face was not visibly speaking.

        - An off-camera DOA segment overlapping a linked voice is re-labelled to that face
          (the person turned away or stepped out of frame).
        - Any remaining uncovered voice time becomes a voice-only segment for that face.
        """
        next_id = max((s.seg_id for s in segs), default=0) + 1
        for vs in self._merged_voice_segments():
            k = linker.track_for_voice(vs.voice_id)
            if k is None:
                continue
            covered = [(s.t_start, s.t_end) for s in segs if s.track_id == k]
            for a, b in _subtract([(vs.t_start, vs.t_end)], covered):
                if b - a < 0.4:
                    continue
                claimed = False
                for s in segs:
                    if s.track_id == OFF_CAMERA and _overlap(s.t_start, s.t_end, a, b) >= 0.5 * s.duration:
                        s.track_id, s.source = k, "doa+voice"
                        claimed = True
                if not claimed:
                    segs.append(Segment(next_id, k, a, b, None, vs.confidence, False, source="voice"))
                    next_id += 1
        return sorted(segs, key=lambda s: s.t_start)

    def _merged_voice_segments(self, gap_s: float = 0.5) -> list[VoiceSegment]:
        out: list[VoiceSegment] = []
        for vs in sorted(self.voice_segments, key=lambda v: (v.voice_id, v.t_start)):
            last = out[-1] if out else None
            if last and last.voice_id == vs.voice_id and vs.t_start - last.t_end <= gap_s:
                last.t_end = max(last.t_end, vs.t_end)
                last.confidence = min(last.confidence, vs.confidence)
            else:
                out.append(VoiceSegment(vs.t_start, vs.t_end, vs.voice_id, vs.confidence))
        return out

    @staticmethod
    def _merge_adjacent(segs, gap_s):
        by_track = defaultdict(list)
        for s in segs:
            by_track[s.track_id].append(s)
        merged = []
        for track_segs in by_track.values():
            track_segs.sort(key=lambda s: s.t_start)
            cur = track_segs[0]
            for s in track_segs[1:]:
                if s.t_start - cur.t_end <= gap_s:
                    d0, d1 = max(cur.duration, 1e-3), max(s.duration, 1e-3)
                    azs = [(a, d) for a, d in ((cur.azimuth_deg, d0), (s.azimuth_deg, d1)) if a is not None]
                    cur = Segment(
                        cur.seg_id, cur.track_id, cur.t_start, max(cur.t_end, s.t_end),
                        sum(a * d for a, d in azs) / sum(d for _, d in azs) if azs else None,
                        (cur.confidence * d0 + s.confidence * d1) / (d0 + d1),
                        cur.overlap or s.overlap,
                        "+".join(sorted(set(cur.source.split("+")) | set(s.source.split("+")))))
                else:
                    merged.append(cur)
                    cur = s
            merged.append(cur)
        return sorted(merged, key=lambda s: s.t_start)

    def _attach_words(self, segs, linker, pad=0.3, max_dist=1.0):
        """Word -> segment. Priority: beam track_id > word voice_id (linked) >
        a single linked voice active at that time > best covering segment > nearest segment."""
        by_track = defaultdict(list)
        for s in segs:
            by_track[s.track_id].append(s)

        def covering(cands, mid):
            return [s for s in cands if s.t_start - pad <= mid <= s.t_end + pad]

        buckets, unassigned = defaultdict(list), []
        for w in sorted(self.words, key=lambda w: w.t_start):
            mid = 0.5 * (w.t_start + w.t_end)
            track = w.track_id
            if track is None and w.voice_id is not None:
                track = linker.track_for_voice(w.voice_id)
            if track is None:
                active_voices = {v.voice_id for v in self.voice_segments if v.t_start <= mid <= v.t_end}
                linked = {linker.track_for_voice(v) for v in active_voices} - {None}
                if len(linked) == 1:
                    track = linked.pop()
            cands = covering(by_track.get(track, []), mid) if track is not None else []
            if not cands:
                cands = covering(segs, mid)
            if not cands:
                dist = lambda s: min(abs(mid - s.t_start), abs(mid - s.t_end))
                cands = sorted((s for s in segs if dist(s) <= max_dist), key=dist)[:1]
            if cands:
                buckets[max(cands, key=lambda s: s.confidence).seg_id].append(w)
            else:
                unassigned.append(w)
        for s in segs:
            s.text = " ".join(w.text.strip() for w in buckets.get(s.seg_id, []))
        return unassigned

    @staticmethod
    def transcript_text(segs, unassigned) -> str:
        lines = []
        for s in segs:
            where = f"{s.azimuth_deg:+.0f} deg" if s.azimuth_deg is not None else "voice only"
            tags = " [overlap]" if s.overlap else ""
            lines.append(f"[{fmt_ts(s.t_start)} - {fmt_ts(s.t_end)}] {s.speaker} ({where}){tags}: {s.text}")
        if unassigned:
            lines.append("")
            lines.append("[Unattributed] " + " ".join(w.text for w in unassigned))
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class AVSpeakerPipeline:
    """Thread-safe entry points for audio, video, diarization and ASR callbacks."""

    def __init__(self, doa_grid_deg, out_prefix: str = "session", names: dict[int, str] | None = None,
                 fusion_kwargs: dict | None = None, linker_kwargs: dict | None = None,
                 segment_kwargs: dict | None = None):
        self.out_prefix = out_prefix
        self.audio = AudioBuffer(doa_grid_deg)
        self.fusion = SpeakingFusion(self.audio, **(fusion_kwargs or {}))
        self.builder = SegmentBuilder(**(segment_kwargs or {}))
        self.linker = VoiceFaceLinker(**(linker_kwargs or {}))
        self.log = EventLog(f"{out_prefix}_events.jsonl", names)
        self._lock = threading.Lock()
        self._last_t: float | None = None
        self.log.write("session_start", 0.0, wall_clock=time.strftime("%Y-%m-%d %H:%M:%S"))

    def on_audio_frame(self, frame: AudioFrame) -> None:
        self.audio.push(frame)  # AudioBuffer has its own lock; keep audio callbacks fast

    def on_video_frame(self, t: float, faces: list[FaceObs]) -> FrameResult:
        """Call ~0.2-0.3 s after capture so the audio window around t is available."""
        with self._lock:
            fr = self.fusion.update(t, faces)
            dt = 0.04 if self._last_t is None else min(0.2, max(0.0, t - self._last_t))
            self._last_t = t
            self.linker.add_frame(fr, dt)
            for seg in self.builder.update(fr):
                self.log.add_segment(seg, self.linker)
            self.log.log_link_changes(self.linker.process_pending(), t)
            return fr

    def on_voice_segment(self, vs: VoiceSegment) -> None:
        with self._lock:
            self.log.voice_segments.append(vs)
            self.linker.add_voice_segment(vs)
            self.log.log_link_changes(self.linker.process_pending(), vs.t_end)

    def on_words(self, words: list[Word]) -> None:
        with self._lock:
            self.log.words.extend(words)

    def close(self):
        with self._lock:
            for seg in self.builder.flush():
                self.log.add_segment(seg, self.linker)
            self.log.log_link_changes(self.linker.process_pending(force=True), self._last_t or 0.0)
            return self.log.finalize(self.linker, self.out_prefix)


# ---------------------------------------------------------------------------
# Synthetic demo
# ---------------------------------------------------------------------------
def _demo(out_prefix: str = "demo_session") -> None:
    """Two presenters + one audience question, with overlap, a lost face and a false smile.

    Script (seconds):
      Alice (-20 deg) speaks 0.5-10, 20-23 (overlaps Bob), 24-35
      Bob   (+25 deg) speaks 11-23, 40-55; his face is LOST 46-50 (turned to the slides)
      Audience (+70 deg, off camera) asks a question 36-39
      Alice smiles/mouths 41-43 without speaking (ASD false positive)
    Diarizer voice IDs are arbitrary (Alice=VOICE_7, Bob=VOICE_3) and arrive 3 s late.
    """
    rng = np.random.default_rng(7)
    grid = np.arange(-90, 91, 2.0)
    fps, audio_hop, duration = 25, 0.02, 60.0
    talkers = {
        "alice":    dict(az=-20.0, track=1,    voice="VOICE_7", spans=[(0.5, 10.0), (20.0, 23.0), (24.0, 35.0)]),
        "bob":      dict(az=25.0,  track=2,    voice="VOICE_3", spans=[(11.0, 23.0), (40.0, 55.0)]),
        "audience": dict(az=70.0,  track=None, voice="VOICE_9", spans=[(36.0, 39.0)]),
    }
    bob_face_lost, alice_smiles = (46.0, 50.0), (41.0, 43.0)
    diar_latency, diar_chunk = 3.0, 2.0

    def active(name, t):
        return any(a <= t < b for a, b in talkers[name]["spans"])

    events = []  # (processing_time, priority, kind, payload)
    for t in np.arange(0.0, duration, audio_hop):
        on = [n for n in talkers if active(n, t)]
        P = rng.uniform(0.0, 0.05, grid.size)
        for n in on:
            P += np.exp(-0.5 * ((grid - talkers[n]["az"]) / 10.0) ** 2)
        events.append((t, 0, "audio", AudioFrame(float(t), 0.95 if on else 0.05, P,
                                                 0.8 if len(on) >= 2 else 0.05)))
    for t in np.arange(0.0, duration, 1.0 / fps):
        faces = []
        for n in ("alice", "bob"):
            if n == "bob" and bob_face_lost[0] <= t < bob_face_lost[1]:
                continue
            p = 0.85 if active(n, t) else 0.08
            if n == "alice" and alice_smiles[0] <= t < alice_smiles[1]:
                p = 0.75
            p = float(np.clip(p + rng.normal(0, 0.08), 0, 1))
            faces.append(FaceObs(talkers[n]["track"], talkers[n]["az"] + rng.normal(0, 1.5), p))
        events.append((t + 0.25, 1, "video", (float(t), faces)))
    for d in talkers.values():
        for a, b in d["spans"]:
            s = a
            while s < b:
                e = min(b, s + diar_chunk)
                events.append((e + diar_latency, 2, "voice", VoiceSegment(s, e, d["voice"], 0.9)))
                s = e
    events.sort(key=lambda ev: (ev[0], ev[1]))

    words, counter = [], defaultdict(int)
    for n, d in talkers.items():
        for a, b in d["spans"]:
            for t in np.arange(a + 0.1, b - 0.3, 0.4):
                counter[n] += 1
                overlapped = sum(active(m, t) for m in talkers) >= 2
                words.append(Word(float(t), float(t) + 0.3, f"{n}{counter[n]}",
                                  voice_id=None if (n == "audience" or overlapped) else d["voice"],
                                  track_id=d["track"] if overlapped else None))

    pipe = AVSpeakerPipeline(grid, out_prefix=out_prefix, names={1: "Alice", 2: "Bob"})
    correct = total = 0
    for _, _, kind, payload in events:
        if kind == "audio":
            pipe.on_audio_frame(payload)
        elif kind == "voice":
            pipe.on_voice_segment(payload)
        else:
            t, faces = payload
            fr = pipe.on_video_frame(t, faces)
            for f in faces:
                truth = active("alice" if f.track_id == 1 else "bob", t)
                correct += fr.speaking[f.track_id] == truth
                total += 1
    pipe.on_words(words)
    segs, unassigned = pipe.close()

    expected = {"alice": "Alice", "bob": "Bob", "audience": "Off-camera"}
    w_ok = w_tot = 0
    for s in segs:
        for token in s.text.split():
            w_tot += 1
            w_ok += expected[token.rstrip("0123456789")] == s.speaker
    w_tot += len(unassigned)

    print("Voice -> face links:")
    for v, (k, share) in sorted(pipe.linker.links.items()):
        print(f"  {v} -> track {k} ({pipe.log.label(k)}), share {share:.2f}")
    unlinked = sorted(set(pipe.linker.voice_total) - set(pipe.linker.links))
    print(f"  unlinked voices: {unlinked or 'none'}")
    print(f"\nFrame-level speaking accuracy (visible faces): {correct / total:.1%}")
    print(f"Word attribution accuracy: {w_ok}/{w_tot} = {w_ok / max(w_tot, 1):.1%}")
    print("\nSegments:")
    for s in segs:
        az = f"{s.azimuth_deg:+6.1f}" if s.azimuth_deg is not None else "   n/a"
        print(f"  #{s.seg_id:<2} {fmt_ts(s.t_start)}-{fmt_ts(s.t_end)}  {s.speaker:<10} az {az}  "
              f"src={s.source:<12} overlap={s.overlap!s:<5} words={len(s.text.split())}")
    print(f"\nWrote {out_prefix}_events.jsonl, {out_prefix}_transcript.json, {out_prefix}_transcript.txt")


if __name__ == "__main__":
    _demo()
