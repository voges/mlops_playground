import os
import torch
import lightning as L
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


# Encoder class that takes an input image and encodes it into a lower-dimensional
# representation.
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=3),
        )

    def forward(self, x):
        return self.l1(x)


# Decoder class that takes the encoded representation and reconstructs the original
# image.
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features=3, out_features=64),
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

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)  # Flatten the input image.
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(input=x_hat, target=x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
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
dataset = MNIST(root=os.getcwd(), download=True, transform=transforms.ToTensor())

# Create a data loader for the training dataset.
train_loader = DataLoader(dataset=dataset)

# Create an instance of the autoencoder model.
autoencoder = LitAutoEncoder(encoder=Encoder(), decoder=Decoder())

# Create a trainer and train the autoencoder.
trainer = L.Trainer(max_epochs=10)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)
