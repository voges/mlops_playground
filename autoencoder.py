import os
import torch
import lightning as L
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 10
NUM_TRAIN_SAMPLES = 10000
LATENT_DIM = 3


# Encoder class that takes an input image and encodes it into a lower-dimensional
# representation.
class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=latent_dim),
        )

    def forward(self, x):
        return self.l1(x)


# Decoder class that takes the encoded representation and reconstructs the original
# image.
class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=28 * 28),
        )

    def forward(self, x):
        return self.l1(x)


class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1)  # Flatten the input image.
        z = self.encoder(x)
        x_hat = self.decoder(z)
        train_loss = F.mse_loss(input=x_hat, target=x)
        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
        return train_loss

    def test_step(self, batch):
        x, _ = batch
        x = x.view(x.size(0), -1)  # Flatten the input image.
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(input=x_hat, target=x)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=LEARNING_RATE)
        return optimizer


# Load the MNIST dataset.
# It consists of a collection of handwritten digits from 0 to 9. Each digit is
# represented as a grayscale image of size 28x28 pixels. When the dataset is loaded,
# the images are transformed into tensors using the transforms.ToTensor() function.
# The resulting tensor has a shape of (1, 28, 28), where 1 represents the number of
# color channels (grayscale images have only one channel), and 28 represents the height
# and width of the image.
# The dataset also contains corresponding labels for each image, indicating the
# digit it represents. The labels are integers ranging from 0 to 9.
# Overall, the MNIST dataset consists of a collection of 60,000 training images
# and 10,000 test images, each with a shape of (1, 28, 28).
train_set = MNIST(
    root=os.getcwd(), train=True, download=True, transform=transforms.ToTensor()
)
test_set = MNIST(
    root=os.getcwd(), train=False, download=True, transform=transforms.ToTensor()
)

# Create a random sampler to sample 10,000 images from the training set with
# replacement.
random_sampler = RandomSampler(
    data_source=train_set, replacement=True, num_samples=NUM_TRAIN_SAMPLES
)

# Create data loaders for the training set and the test set.
train_loader = DataLoader(
    dataset=train_set, batch_size=BATCH_SIZE, sampler=random_sampler
)
test_loader = DataLoader(dataset=test_set)

# Create an instance of the autoencoder model.
autoencoder = LitAutoEncoder(encoder=Encoder(), decoder=Decoder())

# Create a trainer and train the autoencoder.
trainer = L.Trainer(max_epochs=MAX_EPOCHS)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)

# Test the model.
trainer.test(model=autoencoder, dataloaders=test_loader)
