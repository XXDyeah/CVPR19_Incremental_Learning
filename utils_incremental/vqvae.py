import torch
import torch.nn as nn


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
