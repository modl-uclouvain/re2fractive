num_to_select=5

def test_random(trial_dataset_split):
    from re2fractive.acquisition.generic import random_selection

    # Need a better name for 'decorated structures'
    decorated, candidates = trial_dataset_split
    selected = random_selection(candidates.data, decorated.data, num_to_select)
    assert len(selected) == num_to_select

def test_exploration(trial_dataset_split):
    from re2fractive.acquisition.generic import exploration

    # Need a better name for 'decorated structures'
    decorated, candidates = trial_dataset_split
    selected = exploration(candidates.data, decorated.data, "refractive_index",num_to_select)
    assert len(selected) == num_to_select

def test_extremise(trial_dataset_split):
    from re2fractive.acquisition.generic import extremise_expected_value

    decorated, candidates = trial_dataset_split
    selected = extremise_expected_value(candidates.data, decorated.data, "band_gap", num_to_select=num_to_select)
    assert len(selected) == num_to_select

def test_extremise_std(trial_dataset_split):
    from re2fractive.acquisition.generic import extremise_expected_value

    decorated, candidates = trial_dataset_split
    selected = extremise_expected_value(candidates.data, decorated.data, "band_gap", include_std=True, num_to_select=num_to_select)
    assert len(selected) == num_to_select


def test_rppf(trial_dataset_split):
    from re2fractive.acquisition.generic import rppf

    decorated, candidates = trial_dataset_split
    selected = rppf(candidates.data, decorated.data, {"band_gap": "max", "refractive_index": "max"}, num_to_select=num_to_select)
    assert len(selected) == num_to_select

def test_rppf_w_std(trial_dataset_split):
    from re2fractive.acquisition.generic import rppf

    decorated, candidates = trial_dataset_split
    selected = rppf(candidates.data, decorated.data, {"band_gap": "max", "refractive_index": "max"}, include_std=True, num_to_select=num_to_select)
    assert len(selected) == num_to_select


def test_fom_leakage(trial_dataset_split):
    from re2fractive.acquisition.optics import fom_high_k_leakage

    decorated, candidates = trial_dataset_split
    selected = fom_high_k_leakage(candidates.data, decorated.data, include_std=True, num_to_select=num_to_select)
    assert len(selected) == num_to_select

def test_fom_energy(trial_dataset_split):
    from re2fractive.acquisition.optics import fom_high_k_energy

    decorated, candidates = trial_dataset_split
    selected = fom_high_k_energy(candidates.data, decorated.data, include_std=True, num_to_select=num_to_select)
    assert len(selected) == num_to_select
