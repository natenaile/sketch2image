# Bringing Sketches to Life: fast vs full SDXL with ControlNet

This project turns hand-drawn sketches into images with **SDXL + a scribble ControlNet**, and compares two ways of sampling the same model:

- **Fast:** SDXL-Lightning, a distilled model that needs 2–8 steps, a few seconds per image.
- **Quality:** full SDXL with 30 steps and classifier-free guidance.

The question is how much sketch preservation and style control is lost when the model is about 7× faster.

```
sketch2image/
├── sketch2image/
│   ├── preprocess.py   # photo of a pencil sketch -> clean white-on-black ControlNet map
│   ├── pipeline.py     # SDXL + ControlNet, Lightning LoRAs as switchable adapters, baselines
│   ├── metrics.py      # sketch recall / precision / Chamfer (edge-based), CLIP score
│   └── viz.py          # report figures (grids with grouped headers, sweep plot)
├── configs/experiments.yaml   # sketch descriptions, styles, all experiment settings
├── run_experiments.py         # runs everything, writes report-ready figures + tables
├── app.py                     # Gradio web app (draw/upload -> image, fast or quality mode)
├── notebooks/colab_demo.ipynb # one-click run on a free Colab T4
├── sketches/                  # image1..image4: my own hand-drawn sketches
└── tests/                     # CPU tests for preprocessing and the metric
```

## 1. Method

**Preprocessing.** ControlNet expects white strokes on black. A phone photo of a pencil drawing is converted in five steps:
1. grayscale;
2. flatten the lighting (divide by a large Gaussian blur);
3. adaptive threshold;
4. remove specks smaller than 20 px;
5. crop with a 6 % border, resize to 1024 px on the long side (both sides multiples of 8), and dilate the strokes by 2 px.

**Generation.** All outputs come from one copy of the SDXL weights:

| Component | Model |
|---|---|
| Base model | [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |
| VAE (fp16-safe) | [`madebyollin/sdxl-vae-fp16-fix`](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) |
| ControlNet | [`xinsir/controlnet-scribble-sdxl-1.0`](https://huggingface.co/xinsir/controlnet-scribble-sdxl-1.0) |
| Fast sampling | [`ByteDance/SDXL-Lightning`](https://huggingface.co/ByteDance/SDXL-Lightning) 2/4/8-step LoRAs, Euler scheduler with "trailing" timesteps, CFG 0 |
| Full sampling | LoRA disabled, UniPC scheduler, 30 steps, CFG 7 |
| Evaluation | [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) |

The three Lightning LoRAs are loaded as named adapters and switched on and off for each image. The fast and full results therefore differ only in sampling, not in the base model or the ControlNet.

**How the sketch steers the image.** ControlNet is a trainable copy of the UNet encoder that reads the sketch map. At every denoising step, its features are added to the frozen UNet through zero-initialised 1×1 convolutions. This pushes the image's edges onto the strokes. The text prompt still sets colour, material, light and style. The **conditioning scale** sets how strongly the sketch is enforced. **control_end** sets the fraction of steps during which ControlNet is active.

**Metrics.**
- *Sketch recall*: the share of sketch-skeleton pixels that have a Canny edge in the output within 2 % of the image size.
- *Edge precision*: the share of output edges that lie on the sketch. A low value means detail was added.
- *Chamfer distance*: the mean pixel distance from the strokes to the nearest output edge.
- *CLIP score*: 100 × the cosine similarity between the image and the prompt.

## 2. Experiments (88 images on 4 sketches)

| Report item | Question | Settings |
|---|---|---|
| Fig. 2 + Table 2 | Does each mode keep the sketch and follow the style? | 4 styles × {Lightning 4-step, full SDXL} |
| Table 1 | Speed vs quality | Lightning 2 / 4 / 8 steps vs full 30 steps |
| Fig. 4 + Table 3 | Does the sketch matter? | text only, img2img (0.6, 0.85), ControlNet fast / full |
| Fig. 3 + Table 4 | Fixing over-conditioning (pencil lines copied as outlines) | strength 1.0 vs 0.6 vs ControlNet stopped at 50 % |
| Fig. 5 | Effect of sketch strength | conditioning scale 0.2 → 1.5 |
| Text | Seed variability | 3 seeds |

## 3. Install and run

The easiest way is **Colab**. Open `notebooks/colab_demo.ipynb`, choose a T4 GPU, upload this repo as a zip and your sketches `image1`–`image4`, describe them, and run all cells. It takes about 40 minutes, including about 10 GB of model downloads.

To run locally (needs a CUDA GPU with at least 12 GB):
```bash
pip install -r requirements.txt
python tests/test_preprocess_metrics.py        # CPU check, no downloads
python run_experiments.py --preprocess-only    # check the cleaned sketches
python run_experiments.py                      # everything; resumes if interrupted
python app.py --share                          # web app
```
Outputs: `outputs/report/` contains `fig1`–`fig5`, `tables.md` (Tables 1–4 plus the seed sentence), the CSV files and `settings.md` (GPU and library versions). `outputs/all_results.csv` has one row per image with all settings and metrics. Every image uses a fixed seed (42; seeds 1 and 2 for the variability test).

## 4. References

**Models:** listed in the table in section 1. SDXL and SDXL-Lightning are under the OpenRAIL++ license; xinsir ControlNet under Apache-2.0.
**Libraries:** [diffusers](https://github.com/huggingface/diffusers), [transformers](https://github.com/huggingface/transformers), [peft](https://github.com/huggingface/peft), PyTorch, [Gradio](https://github.com/gradio-app/gradio), OpenCV, scikit-image, pandas, matplotlib.

**Papers**
- Podell et al., *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis*, ICLR 2024.
- Lin, Wang & Yang, *SDXL-Lightning: Progressive Adversarial Diffusion Distillation*, arXiv:2402.13929, 2024.
- Zhang, Rao & Agrawala, *Adding Conditional Control to Text-to-Image Diffusion Models* (ControlNet), ICCV 2023.
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022.
- Ho & Salimans, *Classifier-Free Diffusion Guidance*, 2022.
- Meng et al., *SDEdit*, ICLR 2022.
- Zhao et al., *UniPC*, NeurIPS 2023.
- Hu et al., *LoRA*, ICLR 2022.
- Radford et al., *CLIP*, ICML 2021.
- Hessel et al., *CLIPScore*, EMNLP 2021.

**Sketches:** `image1`–`image4` were drawn by me for this project.
**AI assistance:** parts of the code and documentation were drafted with an AI assistant (Claude, Anthropic). I ran all experiments, checked the code and results, and can explain every component.
