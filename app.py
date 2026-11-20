"""Gradio web app: draw or upload a sketch, describe it, pick a style and a speed mode.

  python app.py --share     # public link (works from Colab)
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gradio as gr
import numpy as np

from sketch2image.metrics import sketch_fidelity
from sketch2image.pipeline import RESOLUTION, GenConfig, SketchSDXL
from sketch2image.preprocess import sketch_to_condition

STYLES = {
    "Photoreal": "professional photograph, realistic materials, natural light, highly detailed, sharp focus",
    "Watercolor": "watercolor painting, loose brush strokes, soft pastel washes, textured paper",
    "Concept art": "digital concept art, cinematic lighting, dramatic atmosphere, highly detailed",
    "3D render": "3D render, Blender, studio lighting, soft shadows, smooth materials",
    "None": "",
}
MODES = {"Fast (Lightning, 4 steps)": ("lightning", 4), "Quality (full SDXL, 30 steps)": ("full", 30)}
GEN: SketchSDXL | None = None


def run(drawn, uploaded, subject, style, mode, strength, stop, seed):
    if uploaded is not None:
        img, digital = uploaded, False
    else:
        img = drawn.get("composite") if isinstance(drawn, dict) else drawn
        digital = True
        if img is None or (isinstance(img, np.ndarray) and img.size == 0):
            raise gr.Error("Draw something or upload a sketch first.")
    if not subject.strip():
        raise gr.Error("Describe what you drew, e.g. 'a cozy wooden cabin'.")
    try:
        cond = sketch_to_condition(img, resolution=RESOLUTION, dilate=2, already_clean=digital)
    except ValueError as e:
        raise gr.Error(str(e))
    variant, steps = MODES[mode]
    prompt = ", ".join(p for p in [subject.strip(), STYLES[style]] if p)
    cfg = GenConfig(prompt=prompt, variant=variant, steps=steps, conditioning_scale=strength,
                    control_end=stop, seed=int(seed))
    out = GEN.generate(cond, cfg)
    return cond, out, {**sketch_fidelity(cond, out), "prompt": prompt}


def build_ui():
    with gr.Blocks(title="Sketch to Image") as demo:
        gr.Markdown("# Bringing Sketches to Life\nSDXL + ControlNet scribble. Draw or upload a sketch, describe it, pick a style.")
        with gr.Row():
            with gr.Column():
                with gr.Tab("Draw"):
                    drawn = gr.Sketchpad(label="Sketch (black strokes)", type="numpy", height=420)
                with gr.Tab("Upload"):
                    uploaded = gr.Image(label="Photo or scan of a sketch", type="numpy")
                subject = gr.Textbox(label="What is it?", placeholder="a cozy wooden cabin by a lake")
                style = gr.Dropdown(list(STYLES), value="Photoreal", label="Style")
                mode = gr.Radio(list(MODES), value=list(MODES)[0], label="Mode")
                with gr.Accordion("Advanced", open=False):
                    strength = gr.Slider(0, 2, 1.0, step=0.05, label="Sketch strength (conditioning scale)")
                    stop = gr.Slider(0.1, 1, 1.0, step=0.05, label="Sketch active until (fraction of steps)")
                    seed = gr.Number(42, label="Seed", precision=0)
                btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                cond_out = gr.Image(label="ControlNet input (cleaned sketch)")
                img_out = gr.Image(label="Generated image")
                metrics = gr.JSON(label="Sketch preservation")
        btn.click(run, [drawn, uploaded, subject, style, mode, strength, stop, seed], [cond_out, img_out, metrics])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    GEN = SketchSDXL()
    build_ui().queue().launch(share=args.share, debug=True)
