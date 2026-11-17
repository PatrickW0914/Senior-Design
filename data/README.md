# Test clips, labels and calibration

Clips live in `data/videos/` and are **not committed** (share them through the team drive).
Their labels live in `data/labels/<clip>.csv` and **are committed**, as are the camera
calibration files `tools/calibrate.py` writes to `data/calibration/` (one per camera and
resolution; `config.yaml` `camera.calibration` names the one in use).

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

A label file with only the header line means "nobody speaks in this clip".
`one_person_nodding_silent.csv` is one: the clip was recorded as nodding without speaking, so it
is the known-negative test for the speaking score. No other clip is labeled yet; until they
are, `eval/speaking_accuracy.py` can only score the podcast clips against their audio, which
says that someone is speaking but not who.

Rules (proposed; settle them after two people label the same clip and compare):

- **Person numbers:** person 1 is the leftmost face in the first frame where everyone is visible.
  Numbers stay with the person for the whole clip, even if they swap places.
- **Start:** the first audible word, or without audio, the first visible lip movement of a turn.
- **End:** the last audible word / lip movement. Pauses shorter than 0.5 s inside a turn do not
  split it.
- **Backchannels** ("mm-hm", laughs) count as speaking only if they last 0.3 s or more.
- **Agreement check:** two people label the same clip without looking at each other's CSV. If
  boundaries differ by more than 0.3 s, tighten these rules before labeling the rest.

## Public datasets

`tools/import_dataset.py` brings three public datasets into this layout. Raw downloads stay in
`data/external/` (not committed); the clips land in `data/videos/` with a dataset prefix and their
truth in `data/labels/`. Unlike our own clips these are licensed for reuse with attribution, so
footage from them can appear in a report or demo.

| Dataset | Command | You get | Good for |
|---|---|---|---|
| **AMI Meeting Corpus** (CC BY 4.0) | `python tools/import_dataset.py ami --meeting ES2008a --download --audio` | `ami_<meeting>_closeup1..4.mp4`, one participant each (352x288, 25 fps), with a label file of when that person speaks, from the manual transcript's word timings; `data/audio/ami_<meeting>_array1.wav`, the 8 channels of the table's circular microphone array (10 cm radius, 16 kHz), and `_mix.wav` | Lip-motion accuracy against human transcripts, and later the audio stage: direction of arrival on a real array with known speaker times |
| **AVA-ActiveSpeaker** (labels CC BY 4.0; films from YouTube, mirrored for research) | `python tools/import_dataset.py ava --id 053oq2xB3oU --download --duration 120` | `ava_<id>_<start>s.mp4` and `<clip>.faces.csv`: a box and speaking flag per labeled face per frame | Lip-motion accuracy on varied faces. It is film footage: many faces, a cut every few seconds, so tracks are short |
| **RealVAD** (CC BY 4.0) | `python tools/import_dataset.py realvad --video <file>` | `realvad_panel.mp4` and `.faces.csv`: an upper-body box and speaking flag for each of nine panelists | A fixed camera on seated people, closest to our setup. The video is **not** in the dataset: download `www.youtube.com/watch?v=51pRTOIso4U` yourself. Not yet tested on the real video; check the image the tool writes |

Things to know:

- **Label forms.** AMI close-ups use the same `t_start,t_end,person` file as our own clips. AVA
  and RealVAD use `<clip>.faces.csv` (a box per person per frame), because faces move and shots
  change; a tracked face takes the truth of the labeled box it falls in.
- **AMI close-ups are only "one person" while people sit.** Participants walk to the whiteboard,
  someone else can enter the shot, and seats go empty. The evaluation ignores faces far from
  the seat, but early minutes of a meeting are cleanest. Participants wear headset microphones
  that partly cover the mouth corner.
- **AMI wide views** (`--views corner overhead`) are fisheye overviews with faces about 10 px
  tall: no use for lip reading. Their label file numbers the four speakers A-D as persons 1-4,
  which is not left-to-right order.
- **AMI speech stretches** join words less than 0.5 s apart, the same rule as our own labeling.
  Laughs and coughs in the transcript are not counted as speech.
- **AVA's `speaking`** is true for both `SPEAKING_AUDIBLE` and `SPEAKING_NOT_AUDIBLE`: either way
  the lips move, which is what the visual detector is judged on.
- **RealVAD's readme is wrong about its box format**: it says `x, y, width, height`, but the
  numbers only make sense as two corners `x1, y1, x2, y2` in a 1280x720 frame, which is how the
  converter reads them.
- **EasyCom** was considered and left out: 70 GB as one archive, a head-mounted moving camera,
  and restaurant noise.
- Clips are re-encoded without sound. For AMI use the files in `data/audio/`.
