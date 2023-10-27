import pytest


@pytest.fixture
def trial_dataset():
    from re2fractive.datasets import NaccaratoDataset

    return NaccaratoDataset()


@pytest.fixture
def trial_dataset_split(trial_dataset):
    import random
    import copy

    random.seed(42)
    train_set = copy.copy(trial_dataset)
    split_ind = int(len(train_set) * 0.8)
    train_set.data = train_set.data[:split_ind]

    test_set = copy.copy(trial_dataset)
    test_set.data = test_set.data[split_ind:]
    # Remove the property keys from the test set and use their values plus noise as predictions
    for ind, _ in enumerate(test_set.data):
        test_set.data[ind]["predictions"] = {}
        for property, column in trial_dataset.properties.items():
            value = test_set.data[ind]["attributes"].pop(column)
            noise_mag = 0.1
            noise = random.uniform(-noise_mag, noise_mag)
            test_set.data[ind]["predictions"][property] = value + noise
            test_set.data[ind]["predictions"][f"{property}_std"] = noise_mag

    return train_set, test_set
