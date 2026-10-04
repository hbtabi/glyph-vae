"""Latent-space figures: scatter coloured by digit, decoded manifold, reconstructions, interpolations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .model import ConvVAE  # noqa: E402
from .train import to_numpy_images  # noqa: E402

BG, FG, MUTED = "#0d1117", "#e6edf3", "#8b949e"


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), dpi=fig.dpi)
    plt.close(fig)
    return path


def _project(z: np.ndarray) -> tuple[np.ndarray, str]:
    if z.shape[1] == 2:
        return z, "z"
    zc = z - z.mean(0)
    _, _, vt = np.linalg.svd(zc, full_matrices=False)
    return zc @ vt[:2].T, "PCA of z"


def tile(images: np.ndarray, rows: int, cols: int, pad: int = 1) -> np.ndarray:
    h, w = images.shape[1:]
    out = np.zeros((rows * (h + pad) - pad, cols * (w + pad) - pad), dtype=np.float32)
    for i, im in enumerate(images[: rows * cols]):
        r, c = divmod(i, cols)
        out[r * (h + pad) : r * (h + pad) + h, c * (w + pad) : c * (w + pad) + w] = im
    return out


@torch.no_grad()
def latent_scatter(model: ConvVAE, x: torch.Tensor, y: np.ndarray, path) -> Path:
    mu, _ = model.encode(x)
    z, label = _project(mu.numpy())
    fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=110, facecolor=BG)
    ax.set_facecolor(BG)
    cmap = plt.get_cmap("tab10")
    for d in range(10):
        m = y == d
        ax.scatter(z[m, 0], z[m, 1], s=3, color=cmap(d), alpha=0.6, linewidths=0, label=str(d))
        cx, cy = np.median(z[m], axis=0)
        ax.text(
            cx,
            cy,
            str(d),
            color="white",
            fontsize=14,
            weight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "circle,pad=0.2", "fc": cmap(d), "ec": "none", "alpha": 0.9},
        )
    ax.set_title(f"latent space ({label}), coloured by digit - labels never used in training", color=FG, fontsize=10)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color("#30363d")
    fig.tight_layout()
    return _save(fig, path)


@torch.no_grad()
def manifold(model: ConvVAE, path, n: int = 15, span: float = 2.5) -> Path:
    """Decode a regular grid over the 2 latent dims (others fixed at 0)."""
    g = np.linspace(-span, span, n, dtype=np.float32)
    z = np.zeros((n * n, model.latent_dim), dtype=np.float32)
    for i, yv in enumerate(g[::-1]):
        for j, xv in enumerate(g):
            z[i * n + j, :2] = (xv, yv)
    imgs = to_numpy_images(model.decode(torch.from_numpy(z)))
    fig, ax = plt.subplots(figsize=(6, 6), dpi=110, facecolor=BG)
    ax.imshow(tile(imgs, n, n), cmap="magma", extent=(-span, span, -span, span))
    ax.set_title(f"decoded latent grid z1, z2 in [-{span}, {span}]", color=FG, fontsize=10)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    return _save(fig, path)


@torch.no_grad()
def reconstructions(model: ConvVAE, x: torch.Tensor, path, n: int = 12) -> Path:
    mu, _ = model.encode(x[:n])
    rec = to_numpy_images(model.decode(mu))
    grid = tile(np.concatenate([to_numpy_images(x[:n]), rec]), 2, n)
    fig, ax = plt.subplots(figsize=(n * 0.62, 1.7), dpi=110, facecolor=BG)
    ax.imshow(grid, cmap="magma")
    ax.set_axis_off()
    ax.set_title("top: input   bottom: reconstruction", color=FG, fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


@torch.no_grad()
def interpolation(model: ConvVAE, x: torch.Tensor, y: np.ndarray, pairs: list[tuple[int, int]], path, steps: int = 10) -> Path:
    mu, _ = model.encode(x)
    rows = []
    for a, b in pairs:
        za, zb = mu[np.where(y == a)[0][0]], mu[np.where(y == b)[0][0]]
        alphas = torch.linspace(0, 1, steps).unsqueeze(1)
        rows.append(to_numpy_images(model.decode((1 - alphas) * za + alphas * zb)))
    grid = tile(np.concatenate(rows), len(pairs), steps)
    fig, ax = plt.subplots(figsize=(steps * 0.62, len(pairs) * 0.7 + 0.4), dpi=110, facecolor=BG)
    ax.imshow(grid, cmap="magma")
    ax.set_axis_off()
    ax.set_title("linear interpolation in latent space", color=FG, fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


@torch.no_grad()
def prior_samples(model: ConvVAE, path, n: int = 64, seed: int = 0) -> Path:
    g = torch.Generator().manual_seed(seed)
    imgs = to_numpy_images(model.decode(torch.randn((n, model.latent_dim), generator=g)))
    side = int(np.ceil(np.sqrt(n)))
    fig, ax = plt.subplots(figsize=(5, 5), dpi=110, facecolor=BG)
    ax.imshow(tile(imgs, side, side), cmap="magma")
    ax.set_axis_off()
    ax.set_title("samples from the prior z ~ N(0, I)", color=FG, fontsize=9)
    fig.tight_layout()
    return _save(fig, path)
