# Lip-motion speaking score (2026-11-16)

method `std`, window 0.75 s, threshold 0.03, slope 0.0075, on/off 0.6/0.4. Target: 85% frame-level accuracy on labeled clips.

| clip | truth | judged | accuracy | precision | recall | false-positive rate | fluctuation p50 / p90 |
|---|---|---|---|---|---|---|---|
| ami_es2008a_closeup1 | labels | 7500 face-frames | 96% | 17% | 79% | 4% | 0.008 / 0.022 |
| ami_es2008a_closeup2 | labels | 6329 face-frames | 64% | 22% | 11% | 15% | 0.000 / 0.038 |
| ami_es2008a_closeup3 | labels | 5818 face-frames | 87% | 16% | 51% | 12% | 0.011 / 0.038 |
| ami_es2008a_closeup4 | labels | 7500 face-frames | 89% | 84% | 80% | 7% | 0.003 / 0.093 |
| ava_053oq2xb3ou_900s | face labels | 7830 face-frames | 92% | 64% | 35% | 2% | 0.000 / 0.015 |
| one_person_drinking | none | - | - | - | - | - | 0.004 / 0.020 |
| one_person_hoodie | audio (weak) | 541 frames | 100% | - | - | 0% | 0.004 / 0.009 |
| one_person_nodding_silent | labels (no speech) | 523 face-frames | 86% | 0% | - | 14% | 0.006 / 0.045 |
| one_person_talking | none | - | - | - | - | - | 0.070 / 0.100 |
| one_person_turning | none | - | - | - | - | - | 0.044 / 0.097 |
| one_person_walking | none | - | - | - | - | - | 0.000 / 0.005 |
| three_person_podcast | audio (weak) | 1322 frames | 64% | 97% | 65% | 75% | 0.010 / 0.064 |
| two_person_interview | none | - | - | - | - | - | 0.000 / 0.049 |
| two_person_overlap | none | - | - | - | - | - | 0.016 / 0.041 |
| two_person_podcast | audio (weak) | 1228 frames | 76% | 95% | 78% | 54% | 0.052 / 0.090 |

Truth `labels` = data/labels/<clip>.csv (per person). `audio (weak)` = speech / no speech from the sound track; it cannot say who speaks, so a frame is right when *any* visible face is predicted speaking while there is speech. Lip motion cannot tell talking from chewing or mouthing words; the microphones settle that later.
