# Track IDs (2026-11-13)

Every frame; detector and tracker from config.yaml (max_lost_s = 2.0). Occlusion test: detections suppressed for 1.5 s starting 25%, 50%, 75% into the clip; 'back' = IDs that returned / IDs before, with how far the faces moved meanwhile in face heights.

| clip | people | ids | ids >2s | jumps | longest gap bridged | occlusion: back (moved) | ok? |
|---|---|---|---|---|---|---|---|
| one_person_drinking | 1 | 1 | 1 | 0 | 0 frames (0.00 s) | 1/1 (0.2h), 1/1 (1.0h), 1/1 (1.3h) | yes |
| one_person_hoodie | 1 | 1 | 1 | 0 | 27 frames (0.90 s) | -, 0/1 (1.9h), 1/1 (0.1h) | yes |
| one_person_nodding_silent | 1 | 1 | 1 | 0 | 0 frames (0.00 s) | 1/1 (0.8h), 1/1 (0.3h), 1/1 (1.1h) | yes |
| one_person_talking | 1 | 4 | 4 | 0 | 0 frames (0.00 s) | 1/1 (0.2h), -, - | look |
| one_person_turning | 1 | 8 | 8 | 0 | 48 frames (1.60 s) | 1/1 (0.3h), -, 1/1 (1.0h) | look |
| one_person_walking | 1 | 3 | 3 | 0 | 36 frames (1.20 s) | 1/1 (0.2h), 0/1 (4.3h), 1/1 (0.9h) | look |
| three_person_podcast | 3 | 10 | 10 | 0 | 59 frames (1.97 s) | 2/2 (0.3h), 1/2 (0.9h), 2/2 (0.5h) | look |
| two_person_interview | 2 | 2 | 2 | 0 | 0 frames (0.00 s) | 2/2 (0.1h), 2/2 (0.1h), 2/2 (0.0h) | yes |
| two_person_overlap | 2 | 7 | 2 | 0 | 0 frames (0.00 s) | 2/2 (0.0h), 2/2 (0.1h), 2/2 (0.1h) | yes |
| two_person_podcast | 2 | 1 | 1 | 0 | 1 frames (0.04 s) | 1/1 (0.1h), 1/1 (0.2h), 1/1 (0.1h) | yes |

14.1 min wall.

ids > people means a track was lost for longer than max_lost_s and came back as a new ID (re-ID, Task 12, merges those) or the detector found something that is not one of the people. The podcast clips are edited: a cut can also put a different person where the last one was, and a motion tracker then keeps the old ID (two_person_podcast: 1 ID for 2 people). Only re-ID can tell them apart; the real camera does not cut.
