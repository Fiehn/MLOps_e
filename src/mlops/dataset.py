from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import torchvision.transforms.v2 as transforms

import typer

class MnistDataset(Dataset):
    """MNIST dataset for PyTorch.

    Args:
        data_folder: Path to the data folder.
        train: Whether to load training or test data.
        img_transform: Image transformation to apply.
        target_transform: Target transformation to apply.
    """

    name: str = "MNIST"

    def __init__(
        self,
        data_folder: str = "data/processed",
        train: bool = True,
        img_transform: transforms.Transform | None = None,
        target_transform: transforms.Transform | None = None,
    ) -> None:
        super().__init__()
        self.data_folder = data_folder
        self.train = train
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.load_data()

    def load_data(self) -> None:
        """Load images and targets from disk."""
        images, target = [], []
        if self.train:
            nb_files = len([f for f in os.listdir(self.data_folder) if f.startswith("train_images")])
            for i in range(nb_files):
                images.append(torch.load(f"{self.data_folder}/train_images_{i}.pt"))
                target.append(torch.load(f"{self.data_folder}/train_target_{i}.pt"))
        else:
            images.append(torch.load(f"{self.data_folder}/test_images.pt"))
            target.append(torch.load(f"{self.data_folder}/test_target.pt"))
        self.images = torch.cat(images, 0)
        self.target = torch.cat(target, 0)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return image and target tensor."""
        img, target = self.images[idx], self.target[idx]
        if self.img_transform:
            img = self.img_transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return self.images.shape[0]

def dataset_statistics(datadir: str):
    """Calculate statistics of the dataset."""
    images = torch.load(f"{datadir}/train_images.pt")
    targets = torch.load(f"{datadir}/train_target.pt")
    print(f"Number of images: {images.shape[0]}")
    print(f"Image shape: {images.shape[1:]}")
    print(f"Number of classes: {len(torch.unique(targets))}")

    # train label distribution
    train_label_distribution = torch.bincount(targets)
    print("Train label distribution:")
    print(train_label_distribution)

    # test label distribution
    images = torch.load(f"{datadir}/test_images.pt")
    targets = torch.load(f"{datadir}/test_target.pt")
    test_label_distribution = torch.bincount(targets)
    print("Test label distribution:")
    print(test_label_distribution)

    plt.bar(torch.arange(10), train_label_distribution)
    plt.title("Train label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("train_label_distribution.png")
    plt.close()

    plt.bar(torch.arange(10), test_label_distribution)
    plt.title("Test label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("test_label_distribution.png")
    plt.close()

if __name__ == "__main__":
    typer.run(dataset_statistics)
