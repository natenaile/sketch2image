"""Turn a hand-drawn sketch (phone photo, scan or digital drawing) into a ControlNet condition map.

ControlNet scribble / lineart models were trained on maps with WHITE strokes on a BLACK
background, so a photo of a pencil drawing (dark lines on light paper) has to be
cleaned, binarised and inverted before it is useful as a condition.

Steps
-----
1. Grayscale.
2. Background flattening: divide by a heavily blurred copy of the image. This removes
   shadows and uneven lighting that are typical of phone photos of paper.
3. Adaptive (local) thresholding -> binary stroke mask. Works better than a single global
   threshold when lighting varies across the page.
4. Remove tiny specks (paper texture, noise) with connected-component filtering.
5. Optional dilation to thicken thin pencil lines (scribble models like bold strokes).
6. Crop to the drawing, pad to a margin, resize so the longer side = `resolution` and both
   sides are multiples of 8 (required by the SD VAE, which downsamples by 8).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_image(path_or_img) -> np.ndarray:
    """Load as an RGB uint8 array. Accepts a path, a PIL image or a numpy array.
    Transparent pixels (e.g. from a drawing app) are composited on white."""
    if isinstance(path_or_img, (str, Path)):
        img = Image.open(path_or_img)
    elif isinstance(path_or_img, np.ndarray):
        img = Image.fromarray(path_or_img)
    else:
        img = path_or_img
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return np.array(img.convert("RGB"))


def _to_multiple_of_8(w: int, h: int, resolution: int) -> tuple[int, int]:
    scale = resolution / max(w, h)
    nw = max(8, int(round(w * scale / 8)) * 8)
    nh = max(8, int(round(h * scale / 8)) * 8)
    return nw, nh


def sketch_to_condition(
    image,
    resolution: int = 512,
    dilate: int = 1,
    min_component_area: int = 20,
    margin: float = 0.06,
    already_clean: bool = False,
) -> Image.Image:
    """Return a white-on-black PIL condition image of size (W, H), multiples of 8.

    Parameters
    ----------
    image : path | PIL.Image | np.ndarray
        The sketch. Dark lines on a light background are expected (paper or drawing app).
    resolution : longer side of the output in pixels (512 for SD 1.5).
    dilate : stroke thickening in pixels (0 = none).
    min_component_area : connected components smaller than this (in pixels at the
        original size) are treated as noise and removed.
    margin : fraction of empty border added around the cropped drawing.
    already_clean : skip lighting correction (use for digital drawings).
    """
    rgb = load_image(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if already_clean:
        mask = (gray < 128).astype(np.uint8) * 255
    else:
        # 2. flatten illumination
        k = max(31, (max(gray.shape) // 20) | 1)  # odd kernel, ~5% of image size
        background = cv2.GaussianBlur(gray, (k, k), 0)
        flat = cv2.divide(gray, background, scale=255)
        # 3. adaptive threshold (strokes -> 255)
        block = max(15, (max(gray.shape) // 40) | 1)
        mask = cv2.adaptiveThreshold(
            flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 10
        )

    # 4. remove specks
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_component_area
    mask = (keep[labels] * 255).astype(np.uint8)

    if mask.max() == 0:
        raise ValueError("No strokes found in the sketch. Try a darker pen or `already_clean=True`.")

    # 6a. crop to drawing + margin, keeping it centred on a square-ish canvas
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = mask[y0:y1, x0:x1]
    h, w = crop.shape
    pad = int(margin * max(h, w))
    crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    # 6b. resize to multiple of 8; INTER_AREA keeps thin lines from disappearing
    h, w = crop.shape
    nw, nh = _to_multiple_of_8(w, h, resolution)
    out = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    out = (out > 40).astype(np.uint8) * 255

    # 5. thicken strokes at the final resolution
    if dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        out = cv2.dilate(out, kernel)

    return Image.fromarray(out).convert("RGB")


def condition_to_display(cond: Image.Image) -> Image.Image:
    """Black-on-white version of the condition map, nicer for figures in the report."""
    return Image.fromarray(255 - np.array(cond.convert("L"))).convert("RGB")
