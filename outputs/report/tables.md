# Tables (mean ± std over 4 sketches)

## Table 1. Fast vs full model

| Model | Steps | s / image (median) | Sketch recall ↑ | Chamfer (px) ↓ | CLIP score ↑ |
|---|---|---|---|---|---|
| SDXL-Lightning | 2 | 2.0 | 0.819 ± 0.149 | 6.7 ± 6.3 | 32.5 ± 1.8 |
| SDXL-Lightning | 4 | 3.3 | 0.849 ± 0.162 | 5.5 ± 5.7 | 32.9 ± 1.3 |
| SDXL-Lightning | 8 | 5.9 | 0.809 ± 0.185 | 7.6 ± 8.0 | 33.6 ± 1.7 |
| Full SDXL | 30 | 28.0 | 0.902 ± 0.168 | 4.4 ± 5.9 | 34.3 ± 2.2 |

## Table 2. Style following (CLIP) and sketch preservation (recall) per style

| Style | Lightning CLIP ↑ | Full SDXL CLIP ↑ | Lightning recall ↑ | Full SDXL recall ↑ |
|---|---|---|---|---|
| photoreal | 32.9 ± 1.3 | 34.3 ± 2.2 | 0.849 ± 0.162 | 0.902 ± 0.168 |
| watercolor | 34.0 ± 2.1 | 36.4 ± 2.8 | 0.570 ± 0.292 | 0.659 ± 0.309 |
| concept_art | 35.1 ± 1.0 | 35.9 ± 1.7 | 0.875 ± 0.131 | 0.944 ± 0.071 |
| render3d | 34.4 ± 0.7 | 34.7 ± 2.5 | 0.772 ± 0.216 | 0.863 ± 0.206 |

## Table 3. Does the sketch matter? Baselines vs ControlNet

| Method | Recall, strict ↑ | Recall, lenient (+20 px) | Chamfer (px) ↓ | CLIP score ↑ |
|---|---|---|---|---|
| Text only (no sketch) | 0.687 ± 0.172 | 0.882 ± 0.126 | 11.3 ± 8.2 | 33.4 ± 2.5 |
| img2img, strength 0.6 | 0.983 ± 0.023 | 1.000 ± 0.001 | 4.2 ± 0.9 | 29.3 ± 1.8 |
| img2img, strength 0.85 | 0.945 ± 0.038 | 0.992 ± 0.009 | 4.0 ± 0.5 | 30.3 ± 2.4 |
| ControlNet (Lightning) | 0.849 ± 0.162 | 0.944 ± 0.090 | 5.5 ± 5.7 | 32.9 ± 1.3 |
| ControlNet (full SDXL) | 0.902 ± 0.168 | 0.960 ± 0.079 | 4.4 ± 5.9 | 34.3 ± 2.2 |

## Table 4. Fixing over-conditioning on the densest sketch (image1)

| Setting | Sketch recall | Edge precision | CLIP score |
|---|---|---|---|
| Strength 1.0, ControlNet all steps | 0.822 | 0.977 | 33.0 |
| Strength 0.6 | 0.709 | 0.839 | 33.5 |
| Strength 1.0, ControlNet stopped at 50% | 0.710 | 0.914 | 32.5 |

Same settings, mean over all sketches (for the appendix / repo):

| Setting | Sketch recall | Edge precision | CLIP score |
|---|---|---|---|
| Strength 1.0, ControlNet all steps | 0.849 ± 0.162 | 0.936 ± 0.076 | 32.9 ± 1.3 |
| Strength 0.6 | 0.763 ± 0.185 | 0.863 ± 0.053 | 33.8 ± 1.6 |
| Strength 1.0, ControlNet stopped at 50% | 0.745 ± 0.158 | 0.882 ± 0.062 | 33.8 ± 1.8 |

## Seed variability

Across 3 seeds (SDXL-Lightning, 4 steps), sketch recall varied by ±0.067 and CLIP score by ±1.17 (mean per-sketch standard deviation), so differences smaller than this between settings should not be over-interpreted.

