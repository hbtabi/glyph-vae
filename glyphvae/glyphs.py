"""Procedural "handwritten" digits: no downloads, no MNIST licence questions, infinitely many samples.

Each digit is a set of strokes (polylines / elliptical arcs) on a unit canvas. Every sample gets random
slant, rotation, scale, aspect, translation, per-control-point wobble and pen width, and is rendered with
4x supersampling for anti-aliasing, then downsampled to 28x28.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw


def arc(cx, cy, rx, ry, a0, a1, n=24):
    """Points on an ellipse from angle a0 to a1 (degrees, 0 = right, 90 = down because y points down)."""
    ts = np.radians(np.linspace(a0, a1, n))
    return [(cx + rx * math.cos(t), cy + ry * math.sin(t)) for t in ts]


# Stroke templates on a [0,1]x[0,1] canvas (x right, y down). Each digit can have several styles.
STROKES: dict[int, list[list[list[tuple[float, float]]]]] = {
    0: [[arc(0.5, 0.5, 0.26, 0.38, 0, 360, 40)]],
    1: [[[(0.38, 0.25), (0.55, 0.12), (0.55, 0.88)]], [[(0.52, 0.12), (0.52, 0.88)]]],
    2: [[arc(0.5, 0.33, 0.24, 0.2, 180, 380, 20) + [(0.26, 0.86), (0.76, 0.86)]]],
    3: [[arc(0.48, 0.3, 0.22, 0.18, 200, 450, 22) + arc(0.48, 0.67, 0.26, 0.2, 270, 520, 22)]],
    4: [[[(0.62, 0.88), (0.62, 0.12), (0.22, 0.64), (0.8, 0.64)]], [[(0.3, 0.12), (0.26, 0.56), (0.78, 0.56)], [(0.62, 0.3), (0.62, 0.88)]]],
    5: [[[(0.74, 0.14), (0.32, 0.14), (0.3, 0.46)] + arc(0.49, 0.64, 0.25, 0.22, 230, 500, 24)[1:]]],
    6: [[arc(0.5, 0.42, 0.24, 0.32, 290, 180, 14) + arc(0.5, 0.66, 0.24, 0.2, 180, 540, 30)]],
    7: [[[(0.24, 0.14), (0.78, 0.14), (0.42, 0.88)]], [[(0.24, 0.14), (0.78, 0.14), (0.44, 0.88)], [(0.36, 0.5), (0.68, 0.5)]]],
    8: [[arc(0.5, 0.3, 0.2, 0.17, 0, 360, 30), arc(0.5, 0.67, 0.25, 0.2, 0, 360, 30)]],
    9: [[arc(0.5, 0.34, 0.23, 0.2, 0, 360, 30), [(0.73, 0.34), (0.68, 0.88)]]],
}


def render_digit(d: int, rng: np.random.Generator, size: int = 28, ss: int = 4) -> np.ndarray:
    styles = STROKES[d]
    strokes = styles[rng.integers(len(styles))]
    rot = math.radians(rng.normal(0, 6))
    slant = rng.normal(0.12, 0.12)
    sx = rng.uniform(0.75, 1.0) * rng.uniform(0.85, 1.1)
    sy = rng.uniform(0.8, 1.0)
    tx, ty = rng.normal(0, 0.03, 2)
    wobble = rng.uniform(0.004, 0.018)
    width = rng.uniform(1.6, 3.2) * ss
    W = size * ss
    img = Image.new("L", (W, W), 0)
    draw = ImageDraw.Draw(img)
    cos, sin = math.cos(rot), math.sin(rot)
    for stroke in strokes:
        pts = np.array(stroke, dtype=np.float64) - 0.5
        pts += rng.normal(0, wobble, pts.shape).cumsum(0) * 0.5  # smooth, correlated wobble like a real pen
        x = pts[:, 0] * sx - pts[:, 1] * slant
        y = pts[:, 1] * sy
        x, y = x * cos - y * sin, x * sin + y * cos
        px = (x * 0.82 + 0.5 + tx) * W
        py = (y * 0.82 + 0.5 + ty) * W
        coords = list(zip(px.tolist(), py.tolist(), strict=True))
        draw.line(coords, fill=255, width=int(width), joint="curve")
        r = width / 2
        for cx, cy in (coords[0], coords[-1]):  # round caps
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def make_dataset(n: int, seed: int = 0, size: int = 28) -> tuple[np.ndarray, np.ndarray]:
    """Return (images (n, 1, size, size) in [0,1], labels (n,)), balanced across digits."""
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % 10
    rng.shuffle(labels)
    imgs = np.stack([render_digit(int(d), rng, size) for d in labels])[:, None]
    return imgs.clip(0, 1), labels.astype(np.int64)
