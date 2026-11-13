"""Task 3: speaking-label bookkeeping and the CSV format."""

from labels import Labels, Segment


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
