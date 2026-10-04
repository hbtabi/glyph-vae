"""Convolutional VAE (Kingma & Welling, 2014) with a Bernoulli decoder.

encoder q(z|x) = N(mu(x), diag(sigma^2(x)))       decoder p(x|z) = Bernoulli(logits(z))
loss = BCE(x, x_hat) + beta * KL(q(z|x) || N(0, I))    (the negative ELBO when beta = 1)
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = 2, ch: int = 32, hidden: int = 128):
        super().__init__()
        self.config = {"latent_dim": latent_dim, "ch": ch, "hidden": hidden}
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1),
            nn.SiLU(),  # 28 -> 14
            nn.Conv2d(ch, ch, 4, 2, 1),
            nn.SiLU(),  # 14 -> 7
            nn.Flatten(),
            nn.Linear(ch * 7 * 7, hidden),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.dec_fc = nn.Sequential(nn.Linear(latent_dim, hidden), nn.SiLU(), nn.Linear(hidden, ch * 7 * 7), nn.SiLU())
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, 4, 2, 1),
            nn.SiLU(),  # 7 -> 14
            nn.ConvTranspose2d(ch, 1, 4, 2, 1),  # 14 -> 28 (logits)
        )
        self.ch = ch

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.mu(h), self.logvar(h).clamp(-10, 10)

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(self.dec_fc(z).view(-1, self.ch, 7, 7))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decode_logits(z))

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps lets gradients flow through the sampling step."""
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode_logits(z), mu, logvar

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def vae_loss(
    logits: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-image averages of (total, reconstruction, KL), in nats."""
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / x.shape[0]
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]  # closed form vs N(0, I)
    return recon + beta * kl, recon, kl
