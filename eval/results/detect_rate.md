# Face detection rate (2026-11-13)

Every 5th frame. Reference: SCRFD det_10g, 1280 px wide, confidence 0.3; frontal = yaw ratio <= 0.4; match at IoU >= 0.4. Target: 95% recall on frontal faces.

| clip | backend | recall all | recall frontal | ref faces (frontal) | extra/frame | face h px p10/p50/p90 | ms/frame | 95%? |
|---|---|---|---|---|---|---|---|---|
| one_person_drinking | scrfd | 99% | 99% | 129 (69) | 0.00 | 212/260/380 | 15.9 | yes |
| one_person_drinking | mediapipe | 5% | 9% | 129 (69) | 0.05 | 488/542/583 | 8.9 | NO |
| one_person_hoodie | scrfd | 84% | 94% | 129 (107) | 0.00 | 220/244/341 | 17.3 | NO |
| one_person_hoodie | mediapipe | 0% | 0% | 129 (107) | 0.00 | -/-/- | 9.7 | NO |
| one_person_nodding_silent | scrfd | 99% | 100% | 106 (80) | 0.00 | 294/318/342 | 17.2 | yes |
| one_person_nodding_silent | mediapipe | 1% | 0% | 106 (80) | 0.06 | 486/548/570 | 9.8 | NO |
| one_person_talking | scrfd | 100% | 100% | 866 (827) | 0.00 | 184/189/195 | 17.5 | yes |
| one_person_talking | mediapipe | 0% | 0% | 866 (827) | 0.17 | 248/273/338 | 9.2 | NO |
| one_person_turning | scrfd | 99% | 100% | 1136 (451) | 0.00 | 238/268/282 | 17.6 | yes |
| one_person_turning | mediapipe | 75% | 98% | 1136 (451) | 0.15 | 250/280/315 | 9.3 | yes |
| one_person_walking | scrfd | 97% | 98% | 254 (98) | 0.00 | 218/245/325 | 17.3 | yes |
| one_person_walking | mediapipe | 1% | 2% | 254 (98) | 0.76 | 684/732/824 | 9.9 | NO |
| three_person_podcast | scrfd | 100% | 100% | 603 (522) | 0.00 | 215/233/263 | 15.1 | yes |
| three_person_podcast | mediapipe | 41% | 42% | 603 (522) | 0.12 | 244/264/308 | 4.9 | NO |
| two_person_interview | scrfd | 100% | 100% | 1182 (41) | 0.00 | 182/209/230 | 17.5 | yes |
| two_person_interview | mediapipe | 19% | 32% | 1182 (41) | 0.00 | 189/205/226 | 8.5 | NO |
| two_person_overlap | scrfd | 100% | 100% | 2432 (1970) | 0.00 | 129/136/143 | 16.8 | yes |
| two_person_overlap | mediapipe | 51% | 49% | 2432 (1970) | 0.11 | 160/172/184 | 7.4 | NO |
| two_person_podcast | scrfd | 100% | 100% | 249 (187) | 0.00 | 264/285/300 | 15.5 | yes |
| two_person_podcast | mediapipe | 94% | 96% | 249 (187) | 0.01 | 268/284/318 | 5.1 | yes |
|---|---|---|---|---|---|---|---|---|
| **all clips** | scrfd | 99% | 100% | 7086 (4352) | 0.00 | 132/191/276 | 17.3 | yes |
| **all clips** | mediapipe | 40% | 42% | 7086 (4352) | 0.15 | 166/252/340 | 8.8 | NO |

5502 frames per backend, 14.5 min wall.

Extra boxes are detections that match no reference face: false positives, or faces the reference missed. Face heights are at full resolution.
