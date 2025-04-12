import torch

from src.data import (
    ConcatenatedDataset,
    TrainDataset,
    UnlabeledDataset,
    get_image_paths,
    get_labeled_data_split,
    get_samples,
    get_unlabeled_data_split,
)

TRAIN_FILE_DIR = (
    "/mnt/storage1/shared_data/plant_clef_2025/data/plant_clef_train_281gb/"
)


def test_deterministic_order() -> None:
    # Test to see if multiple calls to get_samples return the same order
    samples = get_samples(TRAIN_FILE_DIR)
    samples2 = get_samples(TRAIN_FILE_DIR)
    assert len(samples) == len(samples2)
    for sample1, sample2 in zip(samples, samples2):
        assert sample1 == sample2


def test_labeled_data_split_size() -> None:
    # Test to see if the data split sizes are correct and unique
    total_samples = len(get_samples(TRAIN_FILE_DIR))
    train_indices, val_indices, test_indices = get_labeled_data_split(TRAIN_FILE_DIR)
    assert len(train_indices) + len(val_indices) + len(test_indices) == total_samples, (
        "Split sizes do not sum to total samples"
    )
    unique_indices = set(train_indices + val_indices + test_indices)
    assert len(unique_indices) == total_samples, "Indices are not unique across splits"


OTHER_TRAIN_FILE_DIR = "/mnt/storage1/shared_data/plant_clef_2025/data/other/"


def test_other_deterministic_order() -> None:
    # Test to see if multiple calls to get_samples return the same order
    samples = get_image_paths(OTHER_TRAIN_FILE_DIR)
    samples2 = get_image_paths(OTHER_TRAIN_FILE_DIR)
    assert len(samples) == len(samples2)
    for sample1, sample2 in zip(samples, samples2):
        assert sample1 == sample2


def test_dataset_concatenation() -> None:
    # Test to see if the concatenated dataset returns the correct number of samples
    train_dataset = TrainDataset(
        image_folder=TRAIN_FILE_DIR,
        image_size=(400, 400),
        transform=None,
        indices=None,
    )
    unlabeled_dataset = UnlabeledDataset(
        image_folder=OTHER_TRAIN_FILE_DIR,
        image_size=(400, 400),
        transform=None,
        indices=None,
    )
    concatenated_dataset = ConcatenatedDataset([train_dataset, unlabeled_dataset])
    assert len(concatenated_dataset) == len(train_dataset) + len(unlabeled_dataset)

    pairs = [
        (concatenated_dataset[0], train_dataset[0]),
        (concatenated_dataset[len(train_dataset)], unlabeled_dataset[0]),
        (concatenated_dataset[-1], unlabeled_dataset[-1]),
        (concatenated_dataset[-len(unlabeled_dataset) - 1], train_dataset[-1]),
        (concatenated_dataset[-len(concatenated_dataset)], train_dataset[0]),
    ]
    for i, (value1, value2) in enumerate(pairs):
        image_1, class_1 = value1
        image_2, class_2 = value2
        assert class_1 == class_2, f"Case {i}"
        assert torch.equal(image_1, image_2), f"Case {i}"


def test_unlabeled_data_split_size() -> None:
    # Test to see if the data split sizes are correct and unique
    total_samples = len(get_samples(TRAIN_FILE_DIR))
    train_indices, val_indices, test_indices = get_unlabeled_data_split(TRAIN_FILE_DIR)
    assert len(train_indices) + len(val_indices) + len(test_indices) == total_samples, (
        "Split sizes do not sum to total samples"
    )
    unique_indices = set(train_indices + val_indices + test_indices)
    assert len(unique_indices) == total_samples, "Indices are not unique across splits"
