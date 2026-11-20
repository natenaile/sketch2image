"""Run every experiment and write the report-ready figures and tables.

  python run_experiments.py                    # everything (resumes if interrupted)
  python run_experiments.py --preprocess-only  # only check the sketch clean-up (no GPU needed)

Inputs : sketches/image1..image4 (.jpg/.jpeg/.png) + their descriptions in configs/experiments.yaml
Outputs: outputs/images/<sketch>/<setting>.png   every generated image
         outputs/log.jsonl                        settings, time and metrics per image (resume log)
         outputs/report/fig1..fig5 *.png          figures in the agreed report layout
         outputs/report/tables.md                 Tables 1-4 + seed sentence, ready to paste
         outputs/report/table*.csv, settings.md   raw numbers and environment for reproducibility
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation on the T4 (set before torch loads)
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from sketch2image.metrics import sketch_fidelity
from sketch2image.pipeline import RESOLUTION, GenConfig
from sketch2image.preprocess import condition_to_display, load_image, sketch_to_condition
from sketch2image.viz import figure_grid, plot_strength_sweep

EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def find_sketches(cfg, sketch_dir: Path) -> dict[str, Path]:
    found = {}
    for name, subject in cfg["sketches"].items():
        path = next((sketch_dir / f"{name}{e}" for e in EXTS if (sketch_dir / f"{name}{e}").exists()), None)
        if path is None:
            print(f"[missing] {sketch_dir}/{name}.jpg/.png not found - skipped")
            continue
        if subject.strip().upper().startswith("DESCRIBE"):
            raise SystemExit(f"Please describe {name} in configs/experiments.yaml (what did you draw?)")
        found[name] = path
    if not found:
        raise SystemExit("No sketches found in sketches/ (expected image1.jpg ... image4.jpg)")
    return found


def build_jobs(cfg, subject: str):
    """List of (experiment, tag, GenConfig). Identical settings share one image (same key)."""
    st, seed, F = cfg["styles"], cfg["seed"], cfg["fast_steps"]
    p = lambda style: f"{subject}, {st[style]}"
    p0 = p("photoreal")
    L = lambda prompt, **kw: GenConfig(prompt=prompt, variant="lightning", steps=kw.pop("steps", F), seed=kw.pop("seed", seed), **kw)
    Fu = lambda prompt, **kw: GenConfig(prompt=prompt, variant="full", steps=cfg["full_steps"],
                                        guidance_scale=cfg["full_guidance"], seed=seed, **kw)
    jobs = []
    for s in st:                                                     # Fig. 2 + Table 2
        jobs += [("styles", f"lightning/{s}", L(p(s))), ("styles", f"full/{s}", Fu(p(s)))]
    for n in cfg["lightning_steps"]:                                 # Table 1
        jobs.append(("steps", f"lightning{n}", L(p0, steps=n)))
    jobs.append(("steps", f"full{cfg['full_steps']}", Fu(p0)))
    jobs.append(("baselines", "Text only (no sketch)", Fu(p0, mode="txt2img")))   # Table 3 + Fig. 4
    for s in cfg["img2img_strengths"]:
        jobs.append(("baselines", f"img2img, strength {s}", Fu(p0, mode="img2img", img2img_strength=s)))
    jobs.append(("baselines", "ControlNet (Lightning)", L(p0)))
    jobs.append(("baselines", "ControlNet (full SDXL)", Fu(p0)))
    oc = cfg["overcond"]                                             # Table 4 + Fig. 3
    jobs += [("overcond", "Strength 1.0, ControlNet all steps", L(p0)),
             ("overcond", f"Strength {oc['reduced_scale']}", L(p0, conditioning_scale=oc["reduced_scale"])),
             ("overcond", f"Strength 1.0, ControlNet stopped at {int(oc['early_stop'] * 100)}%", L(p0, control_end=oc["early_stop"]))]
    for s in cfg["strength_sweep"]:                                  # Fig. 5
        jobs.append(("sweep", f"{s}", L(p0, conditioning_scale=s)))
    for s in [seed] + list(cfg["extra_seeds"]):                      # seed sentence
        jobs.append(("seeds", f"seed {s}", L(p0, seed=s)))
    return jobs


def ms(x, d=3):
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}" if len(x) > 1 else f"{x.mean():.{d}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--sketch-dir", default="sketches")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--preprocess-only", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out)
    rep = out / "report"
    for d in (out / "conditions", rep):
        d.mkdir(parents=True, exist_ok=True)

    # ---------- 1. preprocessing ----------
    sketches = find_sketches(cfg, Path(args.sketch_dir))
    data = {}
    for name, path in sketches.items():
        cond = sketch_to_condition(path, resolution=RESOLUTION, dilate=2)
        disp = condition_to_display(cond)
        cond.save(out / "conditions" / f"{name}_control.png")
        disp.save(out / "conditions" / f"{name}_clean.png")
        density = float((np.array(cond.convert("L")) > 127).mean())
        data[name] = dict(photo=Image.fromarray(load_image(path)), cond=cond, disp=disp, density=density,
                          subject=cfg["sketches"][name])
        print(f"[prep] {name}: {cond.size}, stroke density {density:.3f}")
    names = list(data)
    first = names[0]
    figure_grid([[data[first]["photo"], data[first]["disp"], data[first]["cond"]]], [""],
                ["Photo of sketch", "Cleaned strokes", "ControlNet input"]).save(rep / "fig1_preprocessing.png")
    figure_grid([[data[n]["photo"], data[n]["disp"], data[n]["cond"]] for n in names], names,
                ["Photo of sketch", "Cleaned strokes", "ControlNet input"], cell=220).save(out / "conditions" / "all_preprocessing.png")
    if args.preprocess_only:
        print(f"Preprocessing done -> {out/'conditions'} and {rep/'fig1_preprocessing.png'}")
        return

    # ---------- 2. generation (resumable) ----------
    import pandas as pd
    import torch

    from sketch2image.metrics import ClipScorer
    from sketch2image.pipeline import SketchSDXL

    jobs = {n: build_jobs(cfg, data[n]["subject"]) for n in names}
    log_path = out / "log.jsonl"
    done = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            done[(r["sketch"], r["key"])] = r
    todo, seen = [], set()
    for n in names:
        for _, _, g in jobs[n]:
            k = (n, g.key())
            if k not in seen:
                seen.add(k)
                if k not in done or not (out / "images" / n / f"{g.key()}.png").exists():
                    todo.append((n, g))
    todo.sort(key=lambda t: (t[1].variant != "lightning", t[1].steps))   # fast first; each LoRA loaded once
    print(f"{len(seen)} unique images, {len(seen) - len(todo)} already done, {len(todo)} to generate")

    if todo:
        print("Loading models (first run downloads ~10 GB, takes a few minutes)...")
        gen, clip = SketchSDXL(), ClipScorer()
        t_start = time.time()
        with open(log_path, "a") as log:
            for i, (n, g) in enumerate(todo, 1):
                d = data[n]
                gen.prepare(g)                      # LoRA swap is not part of the timing
                t0 = time.time()
                img = gen.generate(d["cond"], g, sketch_display=d["disp"])
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                sec = time.time() - t0
                (out / "images" / n).mkdir(parents=True, exist_ok=True)
                img.save(out / "images" / n / f"{g.key()}.png")
                row = {"sketch": n, "key": g.key(), "seconds": round(sec, 2), **g.to_dict(),
                       **sketch_fidelity(d["cond"], img), "clip_score": clip(img, g.prompt)}
                log.write(json.dumps(row) + "\n")
                log.flush()
                done[(n, g.key())] = row
                eta = (time.time() - t_start) / i * (len(todo) - i) / 60
                print(f"[{i}/{len(todo)}] {n} {g.key()}: {sec:.1f}s  recall={row['sketch_recall']:.3f} "
                      f"CLIP={row['clip_score']:.1f}  (~{eta:.0f} min left)")

    # ---------- 3. tables ----------
    rows = []
    for n in names:
        for exp, tag, g in jobs[n]:
            rows.append({"experiment": exp, "tag": tag, **done[(n, g.key())]})
    df = pd.DataFrame(rows)
    df.to_csv(out / "all_results.csv", index=False)
    E = lambda e: df[df.experiment == e]
    img = lambda n, key: Image.open(out / "images" / n / f"{key}.png")
    cap = lambda r: f"recall {r['sketch_recall']:.2f} · CLIP {r['clip_score']:.1f}"

    md = [f"# Tables (mean ± std over {len(names)} sketches)\n"]
    # Table 1
    t1 = []
    for tag, sub in E("steps").groupby("tag", sort=False):
        t1.append([("SDXL-Lightning" if tag.startswith("lightning") else "Full SDXL"), int(sub.steps.iloc[0]),
                   f"{sub.seconds.median():.1f}", ms(sub.sketch_recall), ms(sub.chamfer_px, 1), ms(sub.clip_score, 1)])
    md += ["## Table 1. Fast vs full model\n",
           md_table(["Model", "Steps", "s / image (median)", "Sketch recall ↑", "Chamfer (px) ↓", "CLIP score ↑"], t1), ""]
    # Table 2
    S = E("styles").assign(variant_=lambda x: x.tag.str.split("/").str[0], style=lambda x: x.tag.str.split("/").str[1])
    t2 = []
    for style in cfg["styles"]:
        a, b = S[(S["style"] == style) & (S.variant_ == "lightning")], S[(S["style"] == style) & (S.variant_ == "full")]
        t2.append([style, ms(a.clip_score, 1), ms(b.clip_score, 1), ms(a.sketch_recall), ms(b.sketch_recall)])
    md += ["## Table 2. Style following (CLIP) and sketch preservation (recall) per style\n",
           md_table(["Style", "Lightning CLIP ↑", "Full SDXL CLIP ↑", "Lightning recall ↑", "Full SDXL recall ↑"], t2), ""]
    # Table 3
    t3 = [[tag, ms(sub.sketch_recall), ms(sub.chamfer_px, 1), ms(sub.clip_score, 1)]
          for tag, sub in E("baselines").groupby("tag", sort=False)]
    md += ["## Table 3. Does the sketch matter? Baselines vs ControlNet\n",
           md_table(["Method", "Sketch recall ↑", "Chamfer (px) ↓", "CLIP score ↑"], t3), ""]
    # Table 4 (densest sketch) + all-sketch mean
    dense = max(names, key=lambda n: data[n]["density"])
    O = E("overcond")
    t4 = [[r.tag, f"{r.sketch_recall:.3f}", f"{r.edge_precision:.3f}", f"{r.clip_score:.1f}"]
          for r in O[O.sketch == dense].itertuples()]
    t4b = [[tag, ms(sub.sketch_recall), ms(sub.edge_precision), ms(sub.clip_score, 1)] for tag, sub in O.groupby("tag", sort=False)]
    md += [f"## Table 4. Fixing over-conditioning on the densest sketch ({dense})\n",
           md_table(["Setting", "Sketch recall", "Edge precision", "CLIP score"], t4), "",
           "Same settings, mean over all sketches (for the appendix / repo):\n",
           md_table(["Setting", "Sketch recall", "Edge precision", "CLIP score"], t4b), ""]
    # Seeds sentence
    sd = E("seeds").groupby("sketch").agg(r=("sketch_recall", "std"), c=("clip_score", "std"))
    n_seeds = E("seeds").groupby("sketch").size().iloc[0]
    md += ["## Seed variability\n",
           f"Across {n_seeds} seeds (SDXL-Lightning, {cfg['fast_steps']} steps), sketch recall varied by "
           f"±{sd.r.mean():.3f} and CLIP score by ±{sd.c.mean():.2f} (mean per-sketch standard deviation), so "
           f"differences smaller than this between settings should not be over-interpreted.", ""]
    (rep / "tables.md").write_text("\n".join(md) + "\n")
    for i, (h, t) in enumerate([(["model", "steps", "sec", "recall", "chamfer", "clip"], t1),
                                (["style", "light_clip", "full_clip", "light_recall", "full_recall"], t2),
                                (["method", "recall", "chamfer", "clip"], t3),
                                (["setting", "recall", "precision", "clip"], t4)], 1):
        pd.DataFrame(t, columns=h).to_csv(rep / f"table{i}.csv", index=False)

    # ---------- 4. figures ----------
    F, FS = cfg["fast_steps"], cfg["full_steps"]
    look = {(r.sketch, r.tag, r.experiment): r._asdict() for r in df.itertuples()}
    R = lambda n, exp, tag: look[(n, tag, exp)]
    t_fast = E("steps")[E("steps").tag == f"lightning{F}"].seconds.median()
    t_full = E("steps")[E("steps").tag == f"full{FS}"].seconds.median()

    # Fig. 2: rows = sketches; sketch | control | Lightning photo/water | Full photo/water
    imgs, caps = [], []
    for n in names:
        cells = [f"lightning/photoreal", f"lightning/watercolor", f"full/photoreal", f"full/watercolor"]
        rs = [R(n, "styles", c) for c in cells]
        imgs.append([data[n]["photo"], data[n]["cond"]] + [img(n, r["key"]) for r in rs])
        caps.append(["", ""] + [cap(r) for r in rs])
    figure_grid(imgs, names, ["sketch", "control map", "photoreal", "watercolor", "photoreal", "watercolor"], caps,
                groups=[("Your input", "", 0, 2),
                        ("Fast: SDXL-Lightning", f"{F} steps · ~{t_fast:.0f} s / image", 2, 2),
                        ("Quality: full SDXL", f"{FS} steps · ~{t_full:.0f} s / image", 4, 2)]).save(rep / "fig2_main_grid.png")

    # Fig. 3: over-conditioning on the densest sketch (+ every sketch for the repo)
    oc_tags = list(O.tag.unique())
    oc_labels = ["strength 1.0 (default)", f"strength {cfg['overcond']['reduced_scale']}",
                 f"ControlNet stops at {int(cfg['overcond']['early_stop'] * 100)}%"]
    for n in names:
        rs = [R(n, "overcond", t) for t in oc_tags]
        fig = figure_grid([[data[n]["disp"]] + [img(n, r["key"]) for r in rs]], [""], ["sketch"] + oc_labels,
                          [[""] + [cap(r) for r in rs]])
        fig.save(rep / ("fig3_overconditioning.png" if n == dense else f"extra_fig3_overconditioning_{n}.png"))

    # Fig. 4: baselines on image1 (+ every sketch for the repo)
    b_tags = list(E("baselines").tag.unique())
    b_labels = ["text only", f"img2img {cfg['img2img_strengths'][0]}", f"img2img {cfg['img2img_strengths'][1]}",
                "ControlNet (fast)", "ControlNet (full)"]
    for n in names:
        rs = [R(n, "baselines", t) for t in b_tags]
        fig = figure_grid([[data[n]["disp"]] + [img(n, r["key"]) for r in rs]], [""], ["sketch"] + b_labels,
                          [[""] + [cap(r) for r in rs]], cell=260)
        fig.save(rep / ("fig4_baselines.png" if n == first else f"extra_fig4_baselines_{n}.png"))

    # Fig. 5: strength sweep
    plot_strength_sweep(E("sweep")[["conditioning_scale", "sketch_recall", "clip_score"]], rep / "fig5_strength_sweep.png")

    # Extra for the repo: all 4 styles x both models
    for v in ("lightning", "full"):
        rs_all = [[R(n, "styles", f"{v}/{s}") for s in cfg["styles"]] for n in names]
        figure_grid([[data[n]["disp"]] + [img(n, r["key"]) for r in rr] for n, rr in zip(names, rs_all)], names,
                    ["sketch"] + list(cfg["styles"]), [[""] + [cap(r) for r in rr] for rr in rs_all],
                    cell=240).save(rep / f"extra_all_styles_{v}.png")

    # ---------- 5. reproducibility info ----------
    import diffusers
    import transformers

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    (rep / "settings.md").write_text(
        f"GPU: {gpu}\ntorch {torch.__version__}, diffusers {diffusers.__version__}, transformers {transformers.__version__}\n"
        f"Resolution {RESOLUTION} px (long side), fp16, seed {cfg['seed']}\n"
        f"Fast: SDXL-Lightning LoRA, {F} steps, Euler (trailing), CFG 0\n"
        f"Full: SDXL base, {FS} steps, UniPC, CFG {cfg['full_guidance']}\n"
        f"Densest sketch (Table 4 / Fig. 3): {dense}\n")
    print(f"\nDone. Report assets in {rep}/ :")
    for f in sorted(rep.iterdir()):
        print("  ", f.name)


if __name__ == "__main__":
    main()
