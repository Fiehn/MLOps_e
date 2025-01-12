from torch.utils.data import Dataset
import torch
from src.mlops.data import corrupt_mnist

def test_data():
    """Test the corrupt_mnist function."""
    train_set, test_set = corrupt_mnist()
    assert isinstance(train_set, Dataset)
    assert isinstance(test_set, Dataset)


def test_data():
    train, test = corrupt_mnist()
    assert len(train) == 30000
    assert len(test) == 5000
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert y in range(10)
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all()
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all()
