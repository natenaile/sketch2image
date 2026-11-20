"""Sketch-to-image with SDXL + ControlNet, in two speed modes (diffusers).

Models
------
* Base: Stable Diffusion XL 1.0 (stabilityai/stable-diffusion-xl-base-1.0), native 1024 px,
  two text encoders (CLIP ViT-L and OpenCLIP ViT-bigG), 128x128x4 latent.
* VAE: madebyollin/sdxl-vae-fp16-fix (the original SDXL VAE overflows to NaN/black in fp16).
* ControlNet: xinsir/controlnet-scribble-sdxl-1.0, trained on lines of many types and widths,
  so it accepts rough hand-drawn strokes.
* Fast mode: ByteDance SDXL-Lightning LoRA (2-, 4- or 8-step). Lightning is a distilled
  version of SDXL (progressive + adversarial distillation) that samples in a few steps; it needs an
  Euler scheduler with "trailing" timesteps and classifier-free guidance switched off (scale 0).
* Quality mode: the same SDXL with the LoRA disabled, UniPC solver, 30 steps, guidance 7.

Both modes share ONE copy of the SDXL weights: the three Lightning LoRAs are loaded as named
adapters and switched on/off per call, so fast and full results differ only in the sampler.

How the sketch steers the image
-------------------------------
ControlNet is a trainable duplicate of the UNet's encoder that reads the sketch map. At every
denoising step its features are added to the frozen UNet's skip connections and middle block via
1x1 convolutions initialised at zero. The sketch thus pushes each step toward images whose
edges sit on the strokes; the prompt still decides colour, material, light and style.
  conditioning_scale : weight of the injected ControlNet features (0 = sketch ignored)
  control_end        : fraction of the steps during which ControlNet is applied
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from PIL import Image

BASE = "stabilityai/stable-diffusion-xl-base-1.0"
VAE = "madebyollin/sdxl-vae-fp16-fix"
CONTROLNET = "xinsir/controlnet-scribble-sdxl-1.0"
LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
LIGHTNING_STEPS = (2, 4, 8)
RESOLUTION = 1024

DEFAULT_NEGATIVE = (
    "lowres, blurry, sketch, pencil drawing, lineart, monochrome, deformed, distorted, "
    "watermark, text, signature, jpeg artifacts"
)


@dataclass
class GenConfig:
    prompt: str
    variant: str = "lightning"        # lightning | full
    mode: str = "controlnet"          # controlnet | txt2img | img2img
    steps: int = 4                    # lightning: 2/4/8 ; full: e.g. 30
    guidance_scale: float = 7.0       # used by "full" only (Lightning runs at 0)
    conditioning_scale: float = 1.0
    control_end: float = 1.0
    img2img_strength: float = 0.75
    seed: int = 42
    negative_prompt: str = DEFAULT_NEGATIVE

    def to_dict(self):
        return asdict(self)

    def key(self) -> str:
        """Unique, readable id used as the image file name (identical settings are generated once)."""
        k = f"{self.variant}{self.steps}_{self.mode}"
        if self.mode == "controlnet":
            k += f"_cs{self.conditioning_scale:g}_end{self.control_end:g}"
        if self.mode == "img2img":
            k += f"_str{self.img2img_strength:g}"
        if self.variant == "full":
            k += f"_cfg{self.guidance_scale:g}"
        # short fingerprint of the prompt so different styles never share a file
        ph = hashlib.md5((self.prompt + "|" + self.negative_prompt).encode()).hexdigest()[:6]
        return k + f"_seed{self.seed}_p{ph}"


class SketchSDXL:
    """Memory plan for a 16 GB T4: SDXL UNet + 2 text encoders + VAE + ControlNet in fp16 (~9.5 GB),
    and only ONE Lightning LoRA in memory at a time (swapped when the step count changes;
    run_experiments orders the jobs so each LoRA is loaded once)."""

    def __init__(self, device: Optional[str] = None):
        import inspect

        import diffusers as D
        from huggingface_hub import hf_hub_download

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        fp16 = {"variant": "fp16"} if self.dtype == torch.float16 else {}

        vae = D.AutoencoderKL.from_pretrained(VAE, torch_dtype=self.dtype)
        cn = D.ControlNetModel.from_pretrained(CONTROLNET, torch_dtype=self.dtype)
        self.cn_pipe = D.StableDiffusionXLControlNetPipeline.from_pretrained(
            BASE, controlnet=cn, vae=vae, torch_dtype=self.dtype, **fp16)
        self.cn_pipe.to(self.device)
        self.cn_pipe.vae.enable_tiling()     # keeps the 1024 px decode inside T4 memory
        self.cn_pipe.set_progress_bar_config(disable=True)

        # txt2img / img2img baselines: built directly from the SAME module objects
        # (no copy, no dtype/device conversion -> no extra GPU memory)
        def share(cls):
            params = inspect.signature(cls.__init__).parameters
            kw = {k: v for k, v in self.cn_pipe.components.items() if k in params}
            if "add_watermarker" in params:
                kw["add_watermarker"] = False
            pipe = cls(**kw)
            pipe.set_progress_bar_config(disable=True)
            return pipe

        self.t2i = share(D.StableDiffusionXLPipeline)
        self.i2i = share(D.StableDiffusionXLImg2ImgPipeline)

        cfg = self.cn_pipe.scheduler.config
        self.sched_fast = D.EulerDiscreteScheduler.from_config(cfg, timestep_spacing="trailing")
        self.sched_full = D.UniPCMultistepScheduler.from_config(cfg)

        # Lightning LoRA files are only downloaded here; loaded on demand
        self._lora_paths = {n: hf_hub_download(LIGHTNING_REPO, f"sdxl_lightning_{n}step_lora.safetensors")
                            for n in LIGHTNING_STEPS}
        self._adapter: Optional[str] = None

    def _drop_adapter(self):
        if self._adapter is not None:
            self.cn_pipe.delete_adapters(self._adapter)
            self._adapter = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def prepare(self, cfg: GenConfig):
        """Load / unload the right LoRA before timing a generation (idempotent)."""
        if cfg.variant == "lightning":
            if cfg.steps not in LIGHTNING_STEPS:
                raise ValueError(f"Lightning supports {LIGHTNING_STEPS} steps, got {cfg.steps}")
            name = f"lightning{cfg.steps}"
            if self._adapter != name:
                self._drop_adapter()
                self.cn_pipe.load_lora_weights(self._lora_paths[cfg.steps], adapter_name=name)
                self.cn_pipe.unet.to(dtype=self.dtype)   # keep the adapter in fp16 too
                self._adapter = name
            self.cn_pipe.set_adapters([name], adapter_weights=[1.0])
        elif cfg.variant == "full":
            self._drop_adapter()                          # frees memory for CFG (2 UNet passes)
        else:
            raise ValueError(f"unknown variant {cfg.variant}")

    def _set_variant(self, pipe, cfg: GenConfig) -> float:
        self.prepare(cfg)
        if cfg.variant == "lightning":
            pipe.scheduler = self.sched_fast
            return 0.0                                    # Lightning is distilled to run without CFG
        pipe.scheduler = self.sched_full
        return float(cfg.guidance_scale)

    @torch.inference_mode()
    def generate(self, condition: Image.Image, cfg: GenConfig, sketch_display: Optional[Image.Image] = None) -> Image.Image:
        """condition: white-on-black sketch map; sketch_display: black-on-white sketch (img2img start image)."""
        pipe = {"controlnet": self.cn_pipe, "txt2img": self.t2i, "img2img": self.i2i}[cfg.mode]
        gs = self._set_variant(pipe, cfg)
        w, h = condition.size
        common = dict(prompt=cfg.prompt, negative_prompt=cfg.negative_prompt, num_inference_steps=int(cfg.steps),
                      guidance_scale=gs, generator=torch.Generator(self.device).manual_seed(int(cfg.seed)))
        if cfg.mode == "controlnet":
            out = pipe(image=condition, width=w, height=h,
                       controlnet_conditioning_scale=float(cfg.conditioning_scale),
                       control_guidance_start=0.0, control_guidance_end=float(cfg.control_end), **common)
        elif cfg.mode == "txt2img":
            out = pipe(width=w, height=h, **common)
        else:
            if sketch_display is None:
                raise ValueError("img2img needs the black-on-white sketch")
            out = pipe(image=sketch_display.resize((w, h)), strength=float(cfg.img2img_strength), **common)
        return out.images[0]
