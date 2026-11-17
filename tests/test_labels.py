"""Task 3: speaking-label bookkeeping and the CSV format."""

from labels import FaceLabel, FaceLabels, Labels, Segment


def test_toggle_makes_a_segment():
    lb = Labels()
    assert lb.toggle(1, 1.0) == "started" and lb.is_on(1)
    assert lb.toggle(1, 3.5).startswith("stopped") and not lb.is_on(1)
    assert lb.segments == [Segment(1.0, 3.5, 1)]


def test_people_can_overlap():
    lb = Labels()
    lb.toggle(1, 0.0)
    lb.toggle(2, 1.0)
    lb.toggle(1, 2.0)
    lb.toggle(2, 3.0)
    assert lb.speaking_at(0.5) == {1}
    assert lb.speaking_at(1.5) == {1, 2}
    assert lb.speaking_at(2.5) == {2}
    assert lb.speaking_at(3.0) == set()          # t_end is exclusive
    assert lb.people() == {1, 2}


def test_stop_before_start_is_discarded():
    lb = Labels()
    lb.toggle(1, 5.0)
    assert lb.toggle(1, 4.0).startswith("discarded")   # seeked back past the start
    assert lb.segments == [] and not lb.is_on(1)


def test_close_all_and_undo():
    lb = Labels()
    lb.toggle(1, 0.0)
    lb.toggle(1, 1.0)
    lb.toggle(2, 2.0)
    assert lb.close_all(4.0) == 1
    assert lb.segments == [Segment(0.0, 1.0, 1), Segment(2.0, 4.0, 2)]
    assert lb.undo() == Segment(2.0, 4.0, 2)
    assert lb.undo() == Segment(0.0, 1.0, 1)
    assert lb.undo() is None


def test_csv_round_trip_is_sorted(tmp_path):
    lb = Labels([Segment(3.0, 4.0, 2), Segment(1.23456, 2.0, 1)])
    path = tmp_path / "labels" / "clip.csv"
    lb.save(path)
    assert path.read_text(encoding="utf-8").splitlines() == [
        "t_start,t_end,person", "1.235,2.000,1", "3.000,4.000,2"]
    assert Labels.load(path).segments == [Segment(1.235, 2.0, 1), Segment(3.0, 4.0, 2)]


def test_face_labels_lookup_and_round_trip(tmp_path):
    face = FaceLabel(1.0, 100, 100, 200, 220, True, 1, "face")
    body = FaceLabel(1.0, 300, 80, 600, 500, False, 2, "body")
    behind = FaceLabel(1.0, 420, 60, 700, 480, True, 3, "body")                 # overlaps the box of person 2
    fl = FaceLabels([face, body, behind, FaceLabel(2.0, 100, 100, 200, 220, False, 1, "face")])
    assert fl.speaking_for(1.0, 150, 160) is True
    assert fl.speaking_for(1.02, 150, 160) is True                              # nearest labeled frame
    assert fl.speaking_for(1.5, 150, 160) is None                               # no labeled frame within tolerance
    assert fl.speaking_for(2.0, 150, 160) is False
    assert fl.speaking_for(1.0, 1000, 600) is None                              # in nobody's box
    assert fl.speaking_for(1.0, 440, 190) is False                              # head of person 2 ...
    assert fl.speaking_for(1.0, 570, 165) is True                               # ... and of person 3, where the boxes overlap
    assert fl.people() == {1, 2, 3} and fl.span() == (1.0, 2.0)
    fl.save(tmp_path / "c.faces.csv")
    again = FaceLabels.load(tmp_path / "c.faces.csv")
    assert sorted(again.rows, key=lambda r: (r.t, r.person)) == sorted(fl.rows, key=lambda r: (r.t, r.person))
    assert FaceLabels().at(1.0) == [] and FaceLabels().span() == (0.0, 0.0)
