import json

import pytest

from glyphvae.cli import BUNDLED, BUNDLED_2D, main

needs_ckpt = pytest.mark.skipif(not BUNDLED.exists(), reason="bundled checkpoint missing")


def test_preview(tmp_path):
    out = tmp_path / "p.png"
    assert main(["preview", "--out", str(out)]) == 0 and out.stat().st_size > 1000


@needs_ckpt
def test_eval_bundled(capsys):
    assert main(["eval"]) == 0
    m = json.loads(capsys.readouterr().out)
    assert m["latent_dim"] == 8 and m["latent_knn_acc"] > 0.9 and m["neg_elbo"] < 100


@pytest.mark.skipif(not BUNDLED_2D.exists(), reason="bundled 2-D checkpoint missing")
def test_eval_bundled_2d(capsys):
    assert main(["eval", "--ckpt", str(BUNDLED_2D)]) == 0
    m = json.loads(capsys.readouterr().out)
    # a 2-D latent is far more lossy: digits overlap, so kNN is much lower (documented in README)
    assert m["latent_dim"] == 2 and m["latent_knn_acc"] > 0.5 and m["neg_elbo"] < 140


@needs_ckpt
def test_plot_bundled(tmp_path):
    assert main(["plot", "--out-dir", str(tmp_path)]) == 0
    for name in ("latent_space", "latent_space_2d", "manifold", "reconstructions", "interpolation", "samples"):
        assert (tmp_path / f"{name}.png").exists()


def test_train_tiny(tmp_path):
    out = tmp_path / "t.pt"
    assert main(["train", "--epochs", "1", "--n-train", "500", "--latent-dim", "3", "--out", str(out)]) == 0 and out.exists()


def test_missing_checkpoint(tmp_path):
    with pytest.raises(SystemExit):
        main(["eval", "--ckpt", str(tmp_path / "none.pt")])
