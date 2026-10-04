import numpy as np

from glyphvae.glyphs import STROKES, make_dataset, render_digit


def test_every_digit_renders_with_ink():
    rng = np.random.default_rng(0)
    for d in STROKES:
        img = render_digit(d, rng)
        assert img.shape == (28, 28) and 0.04 < img.mean() < 0.4, d
        assert img[0].max() < 0.5 and img[-1].max() < 0.5  # stays inside the canvas


def test_dataset_balanced_and_deterministic():
    x, y = make_dataset(200, seed=3)
    assert x.shape == (200, 1, 28, 28) and x.min() >= 0 and x.max() <= 1
    assert np.bincount(y, minlength=10).tolist() == [20] * 10
    x2, y2 = make_dataset(200, seed=3)
    assert np.array_equal(y, y2) and np.allclose(x, x2)


def test_samples_vary():
    x, y = make_dataset(40, seed=0)
    ones = x[y == 1]
    assert not np.allclose(ones[0], ones[1])
