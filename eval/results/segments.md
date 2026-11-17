# Speaking segments against labels (2026-11-16)

fusion mode `video_only`, tau 0.25 s, gap bridged 0.5 s, onset / release correction 0.0 / 0.5 s; boundary tolerance 0.5 s; replayed from cached measurements.

| clip | turns >= 1 s | found | pieces | start within tol (median error) | end within tol (median error) | speech covered: all / while frontal | segments right | segments | extra |
|---|---|---|---|---|---|---|---|---|---|
| ami_es2008a_closeup1 | 1 (+3 short) | 100% | 1.0 | 100% (+0.27 s) | 100% (-0.44 s) | 40% / 40% | 33% | 4 | 3 |
| ami_es2008a_closeup2 | 11 (+1 short) | 45% | 1.2 | 40% (+0.20 s) | 60% (-0.29 s) | 7% / 12% | 24% | 12 | 5 |
| ami_es2008a_closeup3 | 7 (+1 short) | 29% | 1.5 | 0% (+0.86 s) | 0% (+0.44 s) | 11% / 46% | 26% | 12 | 9 |
| ami_es2008a_closeup4 | 8 (+5 short) | 88% | 2.3 | 100% (-0.29 s) | 86% (+0.25 s) | 77% / 85% | 93% | 18 | 1 |
| one_person_nodding_silent | 0 (+0 short) | - | - | - (-) | - (-) | - / - | 0% | 2 | 2 |
| **all** | 27 (+10 short) | 56% | 1.7 | 67% (+0.14 s) | 67% (-0.05 s) | 39% / 60% | 69% | 48 | 20 |

"While frontal" counts only labeled speech during which that person's face was tracked and not turned past speaking.max_yaw. Median errors are signed: positive means the segment boundary comes after the labeled one. Turns under 1 s are listed as short and not judged.
