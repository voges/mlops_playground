import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L


# Encoder class that takes an input image and encodes it into a lower-dimensional
# representation.
class Encoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
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
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=28 * 28),
        )

    def forward(self, x):
        return self.l1(x)


class AutoEncoder(L.LightningModule):
    def __init__(self, lr: float, latent_dim: int) -> None:
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
        self.lr = lr

    def training_step(self, batch):
        x, _ = batch
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
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)
        return optimizer
