import math

import pytest
import torch

from glyphvae.model import ConvVAE, vae_loss
from glyphvae.train import TrainConfig, evaluate, latent_knn_accuracy, load_checkpoint, save_checkpoint, train


def test_shapes():
    m = ConvVAE(latent_dim=3)
    x = torch.rand(5, 1, 28, 28)
    logits, mu, logvar = m(x)
    assert logits.shape == x.shape and mu.shape == (5, 3) and logvar.shape == (5, 3)
    assert m.decode(torch.zeros(2, 3)).shape == (2, 1, 28, 28)


def test_kl_is_zero_for_standard_normal_posterior():
    logits = torch.zeros(4, 1, 28, 28)
    x = torch.zeros(4, 1, 28, 28)
    _, recon, kl = vae_loss(logits, x, torch.zeros(4, 2), torch.zeros(4, 2))
    assert kl.item() == pytest.approx(0.0, abs=1e-6)
    assert recon.item() == pytest.approx(28 * 28 * math.log(2), rel=1e-5)  # BCE of p=0.5


def test_kl_closed_form_matches_monte_carlo():
    torch.manual_seed(0)
    mu, logvar = torch.tensor([[0.7, -0.3]]), torch.tensor([[-0.5, 0.4]])
    _, _, kl = vae_loss(torch.zeros(1, 1, 28, 28), torch.zeros(1, 1, 28, 28), mu, logvar)
    q = torch.distributions.Normal(mu, (0.5 * logvar).exp())
    p = torch.distributions.Normal(torch.zeros(2), torch.ones(2))
    z = q.sample((200_000,))
    mc = (q.log_prob(z) - p.log_prob(z)).sum(-1).mean()
    assert kl.item() == pytest.approx(mc.item(), rel=0.02)


def test_reparameterization_has_gradients():
    mu = torch.zeros(3, 2, requires_grad=True)
    logvar = torch.zeros(3, 2, requires_grad=True)
    ConvVAE.reparameterize(mu, logvar).sum().backward()
    assert mu.grad is not None and logvar.grad is not None


def test_short_training_learns_and_roundtrips(tmp_path):
    cfg = TrainConfig(latent_dim=4, n_train=2000, n_test=400, epochs=3, batch_size=100, seed=1)
    res = train(cfg, log=None)
    first, last = res.history[0]["neg_elbo"], res.history[-1]["neg_elbo"]
    assert last < first and last < 28 * 28 * math.log(2)  # far better than a coin-flip decoder
    path = save_checkpoint(res, tmp_path / "v.pt")
    model, ck = load_checkpoint(path)
    assert ck["train_config"]["latent_dim"] == 4 and len(ck["history"]) == 3
    from glyphvae.glyphs import make_dataset

    x, y = make_dataset(400, seed=5)
    xt = torch.from_numpy(x)
    assert abs(evaluate(model, xt)["neg_elbo"] - evaluate(res.model, xt)["neg_elbo"]) < 2.0  # fp16 storage is fine
    acc = latent_knn_accuracy(model, xt[:300], torch.from_numpy(y[:300]), xt[300:], torch.from_numpy(y[300:]))
    assert acc > 0.18  # clearly above 10% chance even after 3 tiny epochs on 2k images
