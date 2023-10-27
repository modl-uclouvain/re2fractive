def test_random(trial_dataset_split):
    from re2fractive.selection import random_selection

    # Need a better name for 'decorated structures'
    decorated, candidates = trial_dataset_split
    selected = random_selection(candidates.data, decorated.data, 5)
    assert len(selected) == 5


def test_extremise(trial_dataset_split):
    from re2fractive.selection import extremise_expected_value

    decorated, candidates = trial_dataset_split
    selected = extremise_expected_value(
        candidates.data, decorated.data, "band_gap", 5
    )
    assert len(selected) == 5
