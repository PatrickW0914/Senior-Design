# Speaking segments in video-only mode: settings and what to expect (2026-11-16)

Numbers per clip are in `segments.md` (rerun `python eval/segments_vs_labels.py`). This is the
reasoning behind the `fusion` settings in `config.yaml`.

## Why video-only mode exists

`av_event_log.SpeakingFusion` scores each face as `w_asd*ASD + w_doa*DOA*VAD + w_vad*VAD`. With
no audio frames the last two terms are zero, so with the default weights (0.5 / 0.4 / 0.1) no
score can pass 0.5, and it takes 0.6 to start speaking: nobody is ever marked. The fusion now
warns when it sees that happening. `fusion.mode: video_only` sets the weights to 1 / 0 / 0, which
makes the segment builder, the overlap flag and the event log usable before the microphones
exist. Segments carry no text, and there are no off-camera or voice events. To switch later:
`fusion.mode: audio_visual` and feed `pipe.on_audio_frame()`.

## Result on the clear frontal speaker (AMI close-up 4, 5 minutes)

| Measure | Value |
|---|---|
| Turns of 1 s or more found | 7 of 8 (88%) |
| Start within 0.5 s of the transcript | 100% (median 0.29 s early) |
| End within 0.5 s of the transcript | 86% (median 0.25 s late) |
| Her speaking time inside segments | 77%; 85% of the time her face was frontal |
| Segment time that is real speech | 93% |
| Segments with no speech at all | 1 of 18 |

The other three close-ups are far weaker (11-46% of visible speech covered), for the reason
`speaking_notes.md` gives: those participants are turned away or at the whiteboard, and a face
that is not frontal is not scored. The nodding clip yields two segments, both the known false
positives (walking into frame; the mouth really opening).

## Settings, from a sweep over the labeled clips

- **`asd_release_s: 0.5`.** The lip-motion score looks back 0.75 s, so it stays up after the
  person stops. Uncorrected, segment ends were within 0.5 s for only 29% of the clear speaker's
  turns (median 0.45-0.75 s late). Moving ends 0.5 s earlier gives 86%, and since a segment
  must still last `min_duration_s` afterwards, blips that were mostly tail disappear: segments
  with no speech fell from 53 to 20 across the five clips.
- **`asd_onset_s: 0`.** Starts were already on time or slightly early (lips part before the
  first word).
- **`gap_merge_s: 0.5`, and the cost of it.** The score dips mid-sentence and when the speaker
  glances aside, so a long turn comes out in pieces: 2.3 segments per turn for the clear
  speaker. Bridging 1.0-2.0 s gives 1.9 down to 1.3 pieces, but then neighbouring turns fuse
  (her turns are 1.2-2.2 s apart) and boundaries within 0.5 s fall to 43-57%. Since the Task 9
  check is boundary accuracy, the gap stays at 0.5 s. For a summary where fewer, longer segments
  matter more than exact boundaries, raise it.
- **`tau_s: 0.25`.** 0.1 s was worse on starts (71% against 100%).

## What a transcript looks like now

`<prefix>_transcript.json` has `speakers`, `segments` (speaker, track id, start, end, azimuth,
confidence, overlap flag, empty `text`, `voice_id` null) and an empty `unassigned_words`.
`<prefix>_events.jsonl` is `session_start` (with `mode`), one `segment` line as each segment
closes, and `session_end`. Speakers are named from `overlay.names`, else "Speaker A", "Speaker B"
in order of first speaking; a track lost for more than two seconds comes back with a new ID and
therefore a new speaker name until re-ID (Task 12) merges them.

## Limits

- Turns shorter than about 1 s ("yeah", "okay") are mostly missed: 10 of the 37 labeled turns.
- Everything in `speaking_notes.md` applies: frontal faces only, and lip motion is not speech.
- The settings were tuned on AMI footage; our own clips are still unlabeled.
