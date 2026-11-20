"""Quantitative evaluation.

1. Sketch preservation (structure fidelity), computed without any learned model:
   - Thin the sketch strokes to 1-px skeletons S.
   - Extract edges E from the generated image with Canny (auto thresholds from the median).
   - Distance transforms give, for each pixel, the distance to the nearest edge / stroke.
   * sketch_recall@t  : fraction of sketch skeleton pixels with a generated edge within t px.
                        "Did the sketch's lines survive?"  (main fidelity metric, higher=better)
   * chamfer_px       : mean distance (px) from sketch skeleton pixels to the nearest
                        generated edge (lower = better).
   * edge_precision@t : fraction of generated edges lying within t px of a sketch stroke.
                        Low values are not "bad": they mean the model added texture/detail.
   * edge_f1@t        : harmonic mean of the two.
   A tolerance of ~2% of the image size accounts for the model slightly moving lines.

2. Prompt adherence: CLIP score = 100 * cos(CLIP_image(x), CLIP_text(prompt)) (Hessel et al., 2021).
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _skeleton(mask: np.ndarray) -> np.ndarray:
    try:
        from skimage.morphology import skeletonize

        return skeletonize(mask > 0)
    except ImportError:  # opencv-contrib fallback
        return cv2.ximgproc.thinning((mask > 0).astype(np.uint8) * 255) > 0


def auto_canny(img: Image.Image, sigma: float = 0.33) -> np.ndarray:
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    v = float(np.median(gray))
    lo, hi = int(max(0, (1 - sigma) * v)), int(min(255, (1 + sigma) * v))
    return cv2.Canny(gray, lo, hi) > 0


def sketch_fidelity(condition: Image.Image, generated: Image.Image, tol_frac: float = 0.02) -> dict:
    """condition: white-on-black map; generated: RGB output (resized to the condition size)."""
    cond = np.array(condition.convert("L")) > 127
    gen = generated.convert("RGB").resize(condition.size, Image.BICUBIC)
    edges = auto_canny(gen)
    skel = _skeleton(cond)
    tol = max(2.0, tol_frac * max(cond.shape))

    # distance from every pixel to nearest generated edge / sketch stroke
    d_to_edge = cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 5)
    d_to_sketch = cv2.distanceTransform((~cond).astype(np.uint8), cv2.DIST_L2, 5)

    # cap at the image diagonal (distanceTransform returns huge values if there are no edges at all)
    diag = float(np.hypot(*cond.shape))
    sk_d = np.minimum(d_to_edge[skel].astype(np.float64), diag)
    ed_d = np.minimum(d_to_sketch[edges].astype(np.float64), diag)
    recall = float((sk_d <= tol).mean()) if sk_d.size else 0.0
    precision = float((ed_d <= tol).mean()) if ed_d.size else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "sketch_recall": round(recall, 4),
        "edge_precision": round(precision, 4),
        "edge_f1": round(f1, 4),
        "chamfer_px": round(float(sk_d.mean()) if sk_d.size else float("nan"), 2),
        "tol_px": round(tol, 1),
    }


class ClipScorer:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_name)

    def __call__(self, image: Image.Image, text: str) -> float:
        with self.torch.inference_mode():
            inp = self.proc(text=[text], images=image, return_tensors="pt", padding=True, truncation=True).to(self.device)
            out = self.model(**inp)
            im = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            tx = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            return round(100 * float((im * tx).sum()), 2)


def edge_overlay(condition: Image.Image, generated: Image.Image) -> Image.Image:
    """Visual check for the report: generated image with the sketch strokes drawn in red."""
    gen = np.array(generated.convert("RGB").resize(condition.size, Image.BICUBIC)).copy()
    cond = np.array(condition.convert("L")) > 127
    gen = (gen * 0.75).astype(np.uint8)
    gen[cond] = [255, 40, 40]
    return Image.fromarray(gen)
