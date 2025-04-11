from src.data import get_data_split, get_samples

FILE_DIR = "/mnt/storage1/shared_data/plant_clef_2025/data/plant_clef_train_281gb/"


def test_deterministic_order() -> None:
    # Test to see if multiple calls to get_samples return the same order
    samples = get_samples(FILE_DIR)
    samples2 = get_samples(FILE_DIR)
    assert len(samples) == len(samples2)
    for sample1, sample2 in zip(samples, samples2):
        assert sample1 == sample2


def test_data_split_size() -> None:
    # Test to see if the data split sizes are correct and unique
    total_samples = len(get_samples(FILE_DIR))
    train_indices, val_indices, test_indices = get_data_split(FILE_DIR)
    assert len(train_indices) + len(val_indices) + len(test_indices) == total_samples, (
        "Split sizes do not sum to total samples"
    )
    unique_indices = set(train_indices + val_indices + test_indices)
    assert len(unique_indices) == total_samples, "Indices are not unique across splits"
