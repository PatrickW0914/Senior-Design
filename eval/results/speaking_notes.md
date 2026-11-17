# Lip-motion speaking score: how it was tuned and what it gets wrong (2026-11-16)

Numbers per clip are in `speaking.md` (rerun `python eval/speaking_accuracy.py`). This file is
the reasoning behind the settings in `config.yaml` and the known failure cases, for the report.

## Result

| Clip | Truth | Accuracy | Precision | Recall | False-positive rate |
|---|---|---|---|---|---|
| AMI close-up 4: a seated, frontal speaker | manual transcript | **89%** | 84% | 80% | 7% |
| AMI close-up 1: a listener | manual transcript | 96% | - | - | 4% |
| AMI close-up 3: often turned to a neighbour | manual transcript | 87% | - | 51% | 12% |
| AMI close-up 2: mostly at the whiteboard, back to the camera | manual transcript | 64% | - | 11% | 15% |
| AVA film clip: many faces, a cut every few seconds | per-face labels | 92% | 64% | 35% | 2% |
| Nodding clip: no speech | by construction | 86% | - | - | 14% |
| Drinking clip: no speech (assumed, no audio) | none | - | - | - | 4% predicted speaking |

Mean over the six clips with human-made truth: 85.7%. The Task 8 target of 85% frame-level
accuracy is met on the clear frontal speaker (89%). It is not met where the face cannot be
seen: a person turned away or standing at a whiteboard cannot be lip-read, and their speech
counts as missed. Those frames are what the microphones are for.

Our own clips are not in this table because nobody has labeled them yet. The two podcast clips
were scored against their audio only (someone is speaking, not who): 76% and 64%, a weak check.

## Behaviour on our own clips (no labels yet, so no accuracy)

| Clip | Face height | Frames marked speaking | Reads as |
|---|---|---|---|
| one_person_talking (a continuous talk) | 190 px | 87-94% | right |
| two_person_podcast (host on screen) | 286 px | 76% | plausible |
| one_person_turning (lecturer turning to slides) | 270 px | 56-65% | plausible: turned away is not scored |
| two_person_interview | 224 / 186 px | 41% and 2% | the second person is in profile, past the yaw limit, so never scored |
| two_person_overlap | 133 / 139 px | 25% and 22% | possibly too low: faint lip motion on small faces. Label this one first |
| one_person_walking, one_person_hoodie (silent) | 240 px | 0% | right |

## What the tuning showed

All on cached mouth ratios for 15 clips (`--extract`), so each setting took seconds to try.

1. **A fluctuation threshold of 0.030.** At 0.020 the mean accuracy on human truth is 80%; at
   0.030, 85.5%; at 0.050, 89%, but only by firing less: recall on the clear speaker falls from
   81% to 74% and on the podcast host from 79% to 56%. These clips are mostly people listening,
   so raw accuracy rewards a detector that stays quiet. 0.030 is where precision and recall on
   the clear speaker are both above 80%.
2. **Faces turned past yaw 1.0 are not scored** (`max_yaw`). This took close-up 3 from 69% to
   87%. On a profile the landmarker either finds nothing or puts the lip points on the cheek, and
   the garbage looks like motion. The yaw comes from the detector's keypoints (nose offset from
   the eye midpoint, in eye-distances), which stay reliable on profiles; the landmarker's own
   geometry does not (it hallucinated a frontal face on a profile).
3. **A score needs landmarks without holes** (`max_gap_s` 0.2). Scattered detections on a
   half-turned face are noise.
4. **A head-motion gate was tried and left off** (`max_head_motion`). On small faces a head turn
   alone slides the measured ratio (seen on close-up 1: mouth shut, ratio 0.36 to 0.06 as the
   head came round), so refusing a score while the pose changes cuts nodding false positives
   from 14% to 8%. But an animated speaker moves the head as much as someone nodding (median
   spread 0.13 on the podcast host against 0.09 on the nodding clip), and the same gate took the
   host's recall from 79% to 56%. The option is in the code, off by default.
5. **`std` or `band`, 0.75 s or 1.0 s window: no real difference** (within two points).

## False positives, looked at frame by frame

- **Nodding clip, 8.2-9.6 s:** the mouth really opens, wide enough to show teeth, three times in
  half a second. A yawn or mouthed words; either way the lips moved and no visual-only method
  could tell. This is the limitation the task names, not a defect.
- **Nodding clip, 3.6-4.6 s:** the person walks into frame and the head pitches hard while the
  track starts.
- **Drinking clip:** three blips of 0.2-0.3 s as the bottle meets and leaves the lips.
- **AMI close-ups:** silent smiling and reacting (close-up 3's mouth-opening ratio sits near
  0.25 while silent), and head turns on a small face. The transcript's laughs and coughs were
  not counted as speech, and they account for only 2-11% of the false positives.

## Limits to state in the report

- Lip motion cannot tell talking from chewing, yawning, silent laughing or mouthing words.
- It needs a roughly frontal face. Turned away means no score, so no detection.
- Small faces are noisy: at about 100 px tall (AMI, 352x288 video) a closed mouth already
  fluctuates by 0.010-0.015 against a threshold of 0.030. The 4K phone clips (250-320 px faces)
  sit at 0.003-0.007. Bigger faces give a cleaner signal, as Task 4 expected.
- The score lags by up to the window length (0.75 s) at the start and end of speech.
- Cost: about 12 ms per face per frame on a laptop CPU, on top of 17 ms for detection.

## Still to do

- Label our own two-person clips with `tools/label.py` and rerun the evaluation: the settings
  above were tuned on AMI and AVA footage, not on our camera.
- RealVAD would add a fixed-camera panel; the converter is written but untested until someone
  downloads the video (see `data/README.md`).
