"""Training, evaluation and checkpoint I/O."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .glyphs import make_dataset
from .model import ConvVAE, vae_loss


@dataclass
class TrainConfig:
    latent_dim: int = 2
    n_train: int = 30_000
    n_test: int = 3_000
    epochs: int = 12
    batch_size: int = 128
    lr: float = 2e-3
    beta: float = 1.0
    kl_warmup_epochs: int = 2  # anneal beta from 0 -> beta to avoid posterior collapse early on
    seed: int = 0


@dataclass
class TrainResult:
    model: ConvVAE
    config: TrainConfig
    history: list[dict] = field(default_factory=list)
    seconds: float = 0.0


@torch.no_grad()
def evaluate(model: ConvVAE, x: torch.Tensor, beta: float = 1.0, batch: int = 500) -> dict[str, float]:
    model.eval()
    tot = rec = kl = 0.0
    for i in range(0, len(x), batch):
        xb = x[i : i + batch]
        logits, mu, logvar = model(xb)
        t, r, k = vae_loss(logits, xb, mu, logvar, beta)
        tot, rec, kl = tot + t.item() * len(xb), rec + r.item() * len(xb), kl + k.item() * len(xb)
    n = len(x)
    return {"neg_elbo": tot / n, "recon": rec / n, "kl": kl / n}


@torch.no_grad()
def latent_knn_accuracy(model: ConvVAE, x_train, y_train, x_test, y_test, k: int = 5) -> float:
    """How well digit identity is organised in latent space (the VAE never sees labels)."""
    model.eval()
    ztr, _ = model.encode(x_train)
    zte, _ = model.encode(x_test)
    idx = torch.cdist(zte, ztr).topk(k, largest=False).indices
    votes = y_train[idx]
    pred = torch.mode(votes, dim=1).values
    return (pred == y_test).float().mean().item()


def train(cfg: TrainConfig, log=print) -> TrainResult:
    torch.manual_seed(cfg.seed)
    xtr, ytr = make_dataset(cfg.n_train, seed=cfg.seed)
    xte, yte = make_dataset(cfg.n_test, seed=cfg.seed + 10_000)
    xtr, xte = torch.from_numpy(xtr), torch.from_numpy(xte)
    model = ConvVAE(cfg.latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * ((cfg.n_train + cfg.batch_size - 1) // cfg.batch_size))
    history, t0 = [], time.perf_counter()
    steps_per_epoch = (cfg.n_train + cfg.batch_size - 1) // cfg.batch_size
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(cfg.n_train)
        for i in range(steps_per_epoch):
            progress = (epoch * steps_per_epoch + i) / max(1, cfg.kl_warmup_epochs * steps_per_epoch)
            beta = cfg.beta * min(1.0, progress)
            xb = xtr[perm[i * cfg.batch_size : (i + 1) * cfg.batch_size]]
            logits, mu, logvar = model(xb)
            loss, _, _ = vae_loss(logits, xb, mu, logvar, beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        m = evaluate(model, xte, cfg.beta)
        m["epoch"] = epoch + 1
        history.append(m)
        if log:
            log(f"epoch {epoch + 1:2d}/{cfg.epochs}  test -ELBO {m['neg_elbo']:7.2f}  recon {m['recon']:7.2f}  KL {m['kl']:5.2f}")
    return TrainResult(model.eval(), cfg, history, time.perf_counter() - t0)


def save_checkpoint(res: TrainResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.half() for k, v in res.model.state_dict().items()}  # fp16 keeps the bundled file small
    torch.save({"state_dict": state, "model_config": res.model.config, "train_config": asdict(res.config), "history": res.history}, path)
    return path


def load_checkpoint(path: str | Path) -> tuple[ConvVAE, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = ConvVAE(**ck["model_config"])
    model.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    return model.eval(), ck


def to_numpy_images(t: torch.Tensor) -> np.ndarray:
    return t.detach().clamp(0, 1).squeeze(1).cpu().numpy()
