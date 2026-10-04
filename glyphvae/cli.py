"""glyph-vae CLI: preview | train | eval | plot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .glyphs import make_dataset
from .train import TrainConfig, evaluate, latent_knn_accuracy, load_checkpoint, save_checkpoint, train
from .viz import interpolation, latent_scatter, manifold, prior_samples, reconstructions, tile

CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
BUNDLED = CKPT_DIR / "vae_z8.pt"  # best reconstructions / most informative latent
BUNDLED_2D = CKPT_DIR / "vae_z2.pt"  # 2-D latent: can be plotted and walked directly


def _ckpt(arg: str | None) -> Path:
    p = Path(arg) if arg else BUNDLED
    if not p.exists():
        raise SystemExit(f"checkpoint not found: {p} (train one with `glyph-vae train`)")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="glyph-vae", description="Convolutional VAE on procedurally generated digits")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preview", help="save a grid of generated training digits")
    p.add_argument("--out", default="docs/dataset.png")
    t = sub.add_parser("train", help="train a VAE")
    t.add_argument("--latent-dim", type=int, default=2)
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--n-train", type=int, default=30_000)
    t.add_argument("--beta", type=float, default=1.0, help=">1 = beta-VAE (smoother, more disentangled, blurrier)")
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="runs/vae.pt")
    e = sub.add_parser("eval", help="test-set ELBO and latent kNN accuracy")
    e.add_argument("--ckpt")
    pl = sub.add_parser("plot", help="latent scatter, manifold, reconstructions, interpolations, prior samples")
    pl.add_argument("--ckpt")
    pl.add_argument("--ckpt-2d", help="2-D latent checkpoint for the manifold / raw scatter (default: bundled vae_z2.pt)")
    pl.add_argument("--out-dir", default="docs")
    args = ap.parse_args(argv)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())) if args.cmd == "train" else 1)

    if args.cmd == "preview":
        x, y = make_dataset(200, seed=1)
        order = np.argsort(y, kind="stable")
        grid = tile(x[order, 0], 10, 20)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((grid * 255).astype(np.uint8)).resize((grid.shape[1] * 2, grid.shape[0] * 2), Image.Resampling.NEAREST).save(args.out)
        print("preview ->", args.out)
        return 0

    if args.cmd == "train":
        cfg = TrainConfig(latent_dim=args.latent_dim, epochs=args.epochs, n_train=args.n_train, beta=args.beta, lr=args.lr, seed=args.seed)
        res = train(cfg)
        out = save_checkpoint(res, args.out)
        print(f"done in {res.seconds:.0f}s, {res.model.n_params():,} params -> {out} ({out.stat().st_size / 1024:.0f} KB)")
        return 0

    model, ck = load_checkpoint(_ckpt(args.ckpt))
    xte, yte = make_dataset(3000, seed=10_000 + ck["train_config"]["seed"])
    xte_t = torch.from_numpy(xte)
    if args.cmd == "eval":
        xtr, ytr = make_dataset(5000, seed=777)
        m = evaluate(model, xte_t)
        m["latent_knn_acc"] = latent_knn_accuracy(model, torch.from_numpy(xtr), torch.from_numpy(ytr), xte_t, torch.from_numpy(yte))
        print(json.dumps({"latent_dim": model.latent_dim, **{k: round(v, 3) for k, v in m.items()}}, indent=2))
        return 0

    out = Path(args.out_dir)
    figs = [
        latent_scatter(model, xte_t, yte, out / "latent_space.png"),
        reconstructions(model, xte_t, out / "reconstructions.png"),
        interpolation(model, xte_t, yte, [(1, 7), (3, 8), (0, 6), (4, 9)], out / "interpolation.png"),
        prior_samples(model, out / "samples.png"),
    ]
    # The decoded manifold only makes sense for a 2-D latent: use the model itself if it is 2-D,
    # otherwise the companion 2-D checkpoint.
    m2 = model
    if model.latent_dim != 2:
        p2 = Path(args.ckpt_2d) if args.ckpt_2d else BUNDLED_2D
        m2 = load_checkpoint(p2)[0] if p2.exists() else None
        if m2 is not None and m2.latent_dim == 2:
            figs.append(latent_scatter(m2, xte_t, yte, out / "latent_space_2d.png"))
        else:
            m2 = None
    if m2 is not None:
        figs.append(manifold(m2, out / "manifold.png"))
    for f in figs:
        print("figure ->", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
