import torch

from src.data import (
    ConcatenatedDataset,
    MultitileDataset,
    NonPlantDataset,
    PlantDataset,
    get_image_paths,
    get_labeled_data_split,
    get_plant_data_image_info,
    get_unlabeled_data_split,
)

TRAIN_FILE_DIR = (
    "/mnt/storage1/shared_data/plant_clef_2025/data/plant_clef_train_281gb/"
)


def test_deterministic_order() -> None:
    # Test to see if multiple calls to get_samples return the same order
    samples = get_plant_data_image_info(TRAIN_FILE_DIR)
    samples2 = get_plant_data_image_info(TRAIN_FILE_DIR)
    assert len(samples) == len(samples2)
    for sample1, sample2 in zip(samples, samples2):
        assert sample1 == sample2


def test_labeled_data_split_size() -> None:
    # Test to see if the data split sizes are correct and unique
    plant_data_image_info = get_plant_data_image_info(TRAIN_FILE_DIR)
    total_samples = len(plant_data_image_info)
    train_indices, val_indices, test_indices = get_labeled_data_split(
        plant_data_image_info
    )
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
    train_dataset = PlantDataset(
        plant_data_image_info=get_plant_data_image_info(TRAIN_FILE_DIR),
        image_size=(400, 400),
        transform=None,
        indices=None,
    )
    unlabeled_dataset = NonPlantDataset(
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
        (image_1, class_1, name_1) = value1  # type: ignore
        (image_2, class_2, name_2) = value2
        assert class_1 == class_2, f"Case {i}"
        assert torch.equal(image_1, image_2), f"Case {i}"
        assert name_1 == name_2, f"Case {i}"


def test_unlabeled_data_split_size() -> None:
    # Test to see if the data split sizes are correct and unique
    total_samples = len(get_plant_data_image_info(TRAIN_FILE_DIR))
    train_indices, val_indices, test_indices = get_unlabeled_data_split(TRAIN_FILE_DIR)
    assert len(train_indices) + len(val_indices) + len(test_indices) == total_samples, (
        "Split sizes do not sum to total samples"
    )
    unique_indices = set(train_indices + val_indices + test_indices)
    assert len(unique_indices) == total_samples, "Indices are not unique across splits"


def test_combine_species_threshold() -> None:
    # Test to see if combining species with low counts works correctly
    plant_data_image_info = get_plant_data_image_info(
        TRAIN_FILE_DIR, combine_classes_threshold=0
    )
    assert not any(info.species_id == 0 for info in plant_data_image_info)
    for threshold in [1, 10]:
        plant_data_image_info = get_plant_data_image_info(
            TRAIN_FILE_DIR, combine_classes_threshold=threshold
        )
        class_counts: dict[int, int] = {}
        assert any(info.species_id == 0 for info in plant_data_image_info)
        for info in plant_data_image_info:
            class_counts[info.species_id] = class_counts.get(info.species_id, 0) + 1
        assert all(count > threshold for count in class_counts.values()), (
            "Some classes have less than 2 samples"
        )


TEST_FILE_DIR = "/mnt/storage1/shared_data/plant_clef_2025/data/plant_clef_2025_test/"


def test_multitile_dataset_scale() -> None:
    for scale in [1, 2, 5, 9]:
        dataset = MultitileDataset(
            image_folder=TEST_FILE_DIR,
            scale=scale,
        )
        assert len(dataset) > 0, "Test dataset is empty"
        patches, image_path = dataset[0]
        assert patches.shape[-4] == scale**2


def test_multitile_dataset_overlap() -> None:
    dataset = MultitileDataset(
        image_folder=TEST_FILE_DIR,
        scale=2,
        overlap=0.5,
    )
    assert len(dataset) > 0, "Test dataset is empty"
    patches, image_path = dataset[0]
    assert patches.shape[-4] == 3**2


def test_multitile_dataset_size() -> None:
    dataset = MultitileDataset(
        image_folder=TEST_FILE_DIR,
        tile_size=100,
    )
    assert len(dataset) > 0, "Test dataset is empty"
    patches, image_path = dataset[0]
    assert patches.shape[-2:] == (100, 100)

    dataset = MultitileDataset(
        image_folder=TEST_FILE_DIR,
        tile_size=518,
        scale=2,
        overlap=0.5,
    )
    assert len(dataset) > 0, "Test dataset is empty"
    patches, image_path = dataset[0]
    assert patches.shape[-2:] == (518, 518)
