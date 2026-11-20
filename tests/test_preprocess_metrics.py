"""CPU-only sanity tests (no model download): python -m pytest tests/  or  python tests/test_preprocess_metrics.py"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sketch2image.metrics import sketch_fidelity  # noqa: E402
from sketch2image.preprocess import condition_to_display, sketch_to_condition  # noqa: E402


def fake_phone_photo(h=900, w=700, seed=0):
    """Pencil-like house drawing on unevenly lit, noisy paper."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    paper = 230 - 60 * (xx / w) * (yy / h) + rng.normal(0, 6, (h, w))  # shadow gradient + grain
    img = np.clip(paper, 0, 255).astype(np.uint8)
    col = 90
    cv2.rectangle(img, (200, 400), (500, 700), col, 3)
    cv2.line(img, (180, 410), (350, 250), col, 3)
    cv2.line(img, (350, 250), (520, 410), col, 3)
    cv2.rectangle(img, (320, 560), (380, 700), col, 3)
    cv2.circle(img, (260, 480), 30, col, 3)
    return Image.fromarray(img).convert("RGB")


def test_condition_shape_and_content():
    cond = sketch_to_condition(fake_phone_photo())
    w, h = cond.size
    assert w % 8 == 0 and h % 8 == 0 and max(w, h) == 512
    a = np.array(cond.convert("L"))
    frac = (a > 127).mean()
    assert 0.01 < frac < 0.25, frac  # strokes found, shadow not turned into strokes


def test_metric_identity_and_blank():
    cond = sketch_to_condition(fake_phone_photo())
    same = sketch_fidelity(cond, condition_to_display(cond))
    assert same["sketch_recall"] > 0.9, same
    blank = sketch_fidelity(cond, Image.new("RGB", cond.size, (128, 128, 128)))
    assert blank["sketch_recall"] < 0.05, blank


if __name__ == "__main__":
    test_condition_shape_and_content()
    test_metric_identity_and_blank()
    print("all tests passed")
