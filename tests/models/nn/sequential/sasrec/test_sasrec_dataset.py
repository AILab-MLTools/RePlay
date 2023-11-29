import pytest

from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from replay.models.nn.sequential.sasrec import SasPredictionDataset, SasTrainingDataset, SasValidationDataset

torch = pytest.importorskip("torch")


@pytest.mark.torch
@pytest.mark.parametrize(
    "max_len, feature_name, exception, exception_text",
    [
        (8, "fake_user_id", pytest.raises(ValueError), "Label feature name not found in provided schema"),
        (8, "some_item_feature", pytest.raises(ValueError), "Label feature must be categorical"),
        (8, "some_user_feature", pytest.raises(ValueError), "Label feature must be sequential"),
    ],
)
def test_sasrec_training_dataset_exceptions(wrong_sequential_dataset, max_len, feature_name, exception, exception_text):
    with exception as exc:
        SasTrainingDataset(wrong_sequential_dataset, max_len, label_feature_name=feature_name)

    assert str(exc.value) == exception_text


@pytest.mark.torch
def test_sasrec_datasets_length(sequential_dataset):
    assert len(SasTrainingDataset(sequential_dataset, 8)) == 4
    assert len(SasPredictionDataset(sequential_dataset, 8)) == 4
    assert len(SasValidationDataset(sequential_dataset, sequential_dataset, sequential_dataset, 8)) == 4


@pytest.mark.torch
def test_sasrec_training_dataset_getitem(sequential_dataset):
    batch = SasTrainingDataset(sequential_dataset, 8, label_feature_name="item_id", padding_value=-1)[0]

    assert batch.query_id.item() == 0
    assert all(batch.padding_mask == torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.bool))
    assert all(batch.labels_padding_mask == torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.bool))
    assert all(batch.labels == torch.tensor([-1, -1, -1, -1, -1, -1, 0, 1], dtype=torch.long))


@pytest.mark.torch
def test_sasrec_prediction_dataset_getitem(sequential_dataset):
    batch = SasPredictionDataset(sequential_dataset, 8, padding_value=-1)[1]

    assert batch.query_id.item() == 1
    assert all(batch.padding_mask == torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.bool))


@pytest.mark.torch
def test_sasrec_validation_dataset_getitem(sequential_dataset):
    batch = SasValidationDataset(
        sequential_dataset, sequential_dataset, sequential_dataset, 8, label_feature_name="item_id", padding_value=-1
    )[2]

    assert batch.query_id.item() == 2
    assert all(batch.padding_mask == torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.bool))
    assert all(batch.ground_truth == torch.tensor([1, -1, -1, -1, -1, -1], dtype=torch.long))
    assert all(batch.train == torch.tensor([1, -2, -2, -2, -2, -2], dtype=torch.long))
