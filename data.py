import os

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler

from omegaconf import DictConfig


def load_data():
    # Load the MNIST dataset.
    # It consists of a collection of handwritten digits from 0 to 9. Each digit is
    # represented as a grayscale image of size 28x28 pixels. When the dataset is loaded,
    # the images are transformed into tensors using the transforms.ToTensor() function.
    # The resulting tensor has a shape of (1, 28, 28), where 1 represents the number of
    # color channels (grayscale images have only one channel), and 28 represents the
    # height and width of the image.
    # The dataset also contains corresponding labels for each image, indicating the
    # digit it represents. The labels are integers ranging from 0 to 9.
    # Overall, the MNIST dataset consists of a collection of 60,000 training images
    # and 10,000 test images, each with a shape of (1, 28, 28).
    train_set = MNIST(
        root=os.getcwd(), train=True, download=True, transform=transforms.ToTensor()
    )
    test_set = MNIST(
        root=os.getcwd(),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    return train_set, test_set


def create_data_loaders(cfg: DictConfig, train_set, test_set):
    random_sampler = RandomSampler(
        data_source=train_set,
        replacement=True,
        num_samples=cfg.data.num_train_samples,
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.train.batch_size,
        sampler=random_sampler,
        num_workers=cfg.data.num_workers,
    )

    test_loader = DataLoader(dataset=test_set, num_workers=cfg.data.num_workers)

    return train_loader, test_loader
