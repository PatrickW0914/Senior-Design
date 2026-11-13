# Test clips and labels

Clips live in `data/videos/` and are **not committed** (share them through the team drive).
Their labels live in `data/labels/<clip>.csv` and **are committed**.

## Naming

`<people>_<scenario>.mp4`, lower-case snake_case: `one_person_talking.mp4`, `two_person_overlap.mp4`.
The label file takes the clip's name: `data/labels/two_person_overlap.csv`.

## Clips (as of 2026-11-13)

| Clip | Was | Source | Length | Audio | Exercises |
|---|---|---|---|---|---|
| `one_person_talking.mp4` | OnePersonStandingand Talk | screen recording, public talk | 169 s, 1862x1058 | silent track | baseline: one speaker, frontal |
| `one_person_turning.mp4` | onePersonWalkingandTurningTowardSlides | screen recording, public lecture | 218 s, 1864x1040 | silent track | angled / turned-away face |
| `two_person_interview.mp4` | TwoPeoplePresent | screen recording, public interview | 99 s, 1550x884 | silent track | two faces, one in profile |
| `two_person_overlap.mp4` | TwoPeopleTalkOverEachOther | screen recording, public panel | 201 s, 1272x560 | silent track | two speakers talking over each other |
| `one_person_drinking.mp4` | DrinkingWaterUsingaBottle | self-recorded, classroom | 23 s, 4K | none | mouth hidden by a bottle |
| `one_person_walking.mp4` | WalkingAround | self-recorded, classroom | 57 s, 4K | none | profile, leaving and re-entering the frame (re-ID) |
| `one_person_hoodie.mp4` | WearingAHoodie | self-recorded, classroom | 43 s, 4K, HEVC | yes, quiet | hood up: partly hidden face |
| `one_person_nodding_silent.mp4` | noddingheadwithoutspeaking | self-recorded, classroom | 22 s, 4K | none | head motion without speech (false positives) |
| `two_person_podcast.mp4` | samplevideo1withaudio | screen recording, public podcast | 45 s, 1920x886, 27.52 fps | **yes**, speech at -13 dBFS | two speakers, one on screen at a time, 6 camera cuts |
| `three_person_podcast.mp4` | samplevideo2withaudio | screen recording, public podcast | 44 s, 1920x886 | **yes**, speech at -12 dBFS | two or three faces, 8 cuts, a B-roll web page with 8 printed faces at 14-17 s (false positives) |

**Audio:** only the two podcast clips and `one_person_hoodie.mp4` have sound. The podcast audio is
clean speech (floor about -45 dBFS) but **dual mono**: left and right are identical, so it gives
voice activity and a transcript, not a direction. The other screen recordings were captured
without system audio and three phone clips without a microphone, so speaking labels for those can
only come from lip movement. Real direction-of-arrival tests need the mic array.

**Camera cuts:** the podcast clips are edited: each cut is a new shot, so faces appear and vanish
with no motion, and the speaker is usually the one shown. Fine for audio-visual sync and labeling;
not a fair test of tracking or of speaker detection, since the editing already picks the speaker.

**Consent:** the screen recordings show people who have not consented to be in this project.
Fine for internal testing; check before any appear in a report or demo.

## Labeling

```
python tools/label.py --source data/videos/two_person_overlap.mp4
```

Press `1` when person 1 starts speaking and again when they stop; same for `2`. Keys are listed
in the window. The CSV has one row per speaking segment: `t_start,t_end,person`, in seconds of
clip time (frame index / fps), so it lines up with `output/*_faces.jsonl` from `main.py`.

Rules (proposed; settle them after two people label the same clip and compare):

- **Person numbers:** person 1 is the leftmost face in the first frame where everyone is visible.
  Numbers stay with the person for the whole clip, even if they swap places.
- **Start:** the first audible word, or without audio, the first visible lip movement of a turn.
- **End:** the last audible word / lip movement. Pauses shorter than 0.5 s inside a turn do not
  split it.
- **Backchannels** ("mm-hm", laughs) count as speaking only if they last 0.3 s or more.
- **Agreement check:** two people label the same clip without looking at each other's CSV. If
  boundaries differ by more than 0.3 s, tighten these rules before labeling the rest.
