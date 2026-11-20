"""Report figures: labelled image grids with grouped column headers, and the sweep plot."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int, bold: bool = False):
    names = ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Arial Bold.ttf"] if bold else \
            ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"]
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit(im: Image.Image, cell: int) -> Image.Image:
    im = im.convert("RGB").copy()
    im.thumbnail((cell, cell), Image.LANCZOS)
    canvas = Image.new("RGB", (cell, cell), "white")
    canvas.paste(im, ((cell - im.width) // 2, (cell - im.height) // 2))
    return canvas


def figure_grid(images, row_labels, col_labels, captions=None, groups=None, cell=300, gap=8) -> Image.Image:
    """images[r][c]: PIL images (None = empty). captions[r][c]: short text under each image.
    groups: list of (label, sublabel, first_col, n_cols) drawn as a header band over columns."""
    R, C = len(images), len(images[0])
    left = 150 if any(row_labels) else 10
    group_h = 62 if groups else 0
    col_h, cap_h = 30, (26 if captions else 0)
    W = left + C * (cell + gap) + 10
    H = group_h + col_h + R * (cell + cap_h + gap) + 10
    fig = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(fig)
    fb, f, fs = _font(22, True), _font(19), _font(16)
    x0 = lambda c: left + c * (cell + gap)

    for label, sub, c0, n in groups or []:
        xa, xb = x0(c0), x0(c0 + n) - gap
        d.line([(xa, group_h - 8), (xb, group_h - 8)], fill=(90, 90, 90), width=2)
        d.text(((xa + xb) / 2, 6), label, fill="black", font=fb, anchor="ma")
        if sub:
            d.text(((xa + xb) / 2, 33), sub, fill=(90, 90, 90), font=fs, anchor="ma")
    for c, lab in enumerate(col_labels):
        d.text((x0(c) + cell / 2, group_h + 4), lab, fill="black", font=f, anchor="ma")
    for r in range(R):
        y = group_h + col_h + r * (cell + cap_h + gap)
        if row_labels[r]:
            d.text((12, y + cell / 2), row_labels[r], fill="black", font=fb, anchor="lm")
        for c in range(C):
            if images[r][c] is not None:
                fig.paste(_fit(images[r][c], cell), (x0(c), y))
            if captions and captions[r][c]:
                d.text((x0(c) + cell / 2, y + cell + 4), captions[r][c], fill=(60, 60, 60), font=fs, anchor="ma")
    return fig


def plot_strength_sweep(df, out_path: Path):
    """Fig. 5: sketch recall and CLIP score (mean +/- std over sketches) vs conditioning scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = df.groupby("conditioning_scale").agg(["mean", "std"])
    x = g.index.values
    fig, ax1 = plt.subplots(figsize=(6.4, 3.6))
    blue, orange = "#2a6fdb", "#d9622b"
    ax1.errorbar(x, g[("sketch_recall", "mean")], yerr=g[("sketch_recall", "std")], color=blue,
                 marker="o", capsize=3, lw=2)
    ax1.set_ylabel("Sketch recall (structure kept)", color=blue)
    ax1.set_ylim(0, 1.02)
    ax1.set_xlabel("Sketch strength (ControlNet conditioning scale)")
    ax1.tick_params(axis="y", colors=blue)
    ax2 = ax1.twinx()
    ax2.errorbar(x, g[("clip_score", "mean")], yerr=g[("clip_score", "std")], color=orange,
                 marker="s", capsize=3, lw=2)
    ax2.set_ylabel("CLIP score (prompt match)", color=orange)
    ax2.tick_params(axis="y", colors=orange)
    for a in (ax1, ax2):
        a.spines["top"].set_visible(False)
    ax1.grid(alpha=0.25)
    ax1.set_title("Effect of sketch strength (SDXL-Lightning, mean ± std over sketches)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
