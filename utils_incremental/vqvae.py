import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )


class VectorQuantizer(nn.Module):
    """Basic vector quantization layer used by ``VQVAE``."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e: torch.Tensor):
        B, D, H, W = z_e.shape
        z = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)
        distances = (z.pow(2).sum(1, keepdim=True)
                     - 2 * z @ self.codebook.weight.t()
                     + self.codebook.weight.pow(2).sum(1))
        indices = distances.argmin(1)
        z_q = self.codebook(indices).view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        return z_q, indices.view(B, H, W)


class VQVAE(nn.Module):
    """A tiny VQ-VAE for counterfactual sample generation."""

    def __init__(self, in_channels: int = 3, hidden_channels: int = 64,
                 embedding_dim: int = 64, num_embeddings: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hidden_channels),
            nn.Conv2d(hidden_channels, embedding_dim, kernel_size=1)
        )
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1)
        )

    def encode(self, x: torch.Tensor):
        z_e = self.encoder(x)
        z_q, indices = self.quantizer(z_e)
        return z_q, indices

    def decode(self, z_q: torch.Tensor):
        return self.decoder(z_q)


def pretrain_vqvae(model: VQVAE, dataset, device: torch.device,
                   epochs: int = 100, lr: float = 1e-3,
                   sample_size: int = 50000, batch_size: int = 128):
    """Light‑weight pretraining for ``VQVAE``.

    When no checkpoint is provided, we quickly warm up the VQ‑VAE on a
    subset of the available dataset so that counterfactual samples are
    at least meaningful.  The training is intentionally tiny – a few
    thousand images for a couple of epochs – to avoid a heavy cost
    during incremental learning.

    Parameters
    ----------
    model : VQVAE
        Instance to train.
    dataset : torch.utils.data.Dataset
        Source dataset (e.g. CIFAR100 training split).
    device : torch.device
        Device on which to run the training.
    epochs : int, optional
        Number of training epochs, by default 5.
    lr : float, optional
        Learning rate for Adam, by default 1e-3.
    sample_size : int, optional
        Number of images used for the warm‑up, by default 5000.
    batch_size : int, optional
        Mini‑batch size, by default 128.
    """

    model.train()
    subset = Subset(dataset, list(range(min(len(dataset), sample_size))))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        for x, _ in loader:
            x = x.to(device)
            z_e = model.encoder(x)
            z_q, _ = model.quantizer(z_e)
            x_recon = model.decoder(z_q)

            recon_loss = F.mse_loss(x_recon, x)
            commit_loss = F.mse_loss(z_e.detach(), z_q)
            embed_loss = F.mse_loss(z_e, z_q.detach())
            loss = recon_loss + commit_loss + embed_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

