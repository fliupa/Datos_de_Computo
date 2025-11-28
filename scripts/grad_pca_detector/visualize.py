from typing import List, Dict, Optional
from PIL import Image, ImageDraw

def _create_canvas(w: int = 800, h: int = 600, bg: str = "#ffffff"):
    return Image.new("RGB", (w, h), bg)

def _draw_axes(draw: ImageDraw.ImageDraw, w: int, h: int, lm: int, rm: int, tm: int, bm: int):
    draw.line([(lm, h - bm), (w - rm, h - bm)], fill="#000000", width=1)
    draw.line([(lm, tm), (lm, h - bm)], fill="#000000", width=1)

def _map(v: float, vmin: float, vmax: float, pmin: float, pmax: float):
    if vmax == vmin:
        return (pmin + pmax) * 0.5
    t = (v - vmin) / (vmax - vmin)
    return pmin + t * (pmax - pmin)

def plot_scatter(rows: List[Dict], out_path: Optional[str] = None, show: bool = False):
    if not rows:
        return None
    w, h = 800, 600
    lm, rm, tm, bm = 60, 20, 20, 40
    img = _create_canvas(w, h)
    d = ImageDraw.Draw(img)
    _draw_axes(d, w, h, lm, rm, tm, bm)
    xs = [float(r.get("evr1", 0.0)) for r in rows]
    ys = [float(r.get("grad_std", 0.0)) for r in rows]
    xmin, xmax = (min(xs), max(xs)) if xs else (0.0, 1.0)
    ymin, ymax = (min(ys), max(ys)) if ys else (0.0, 1.0)
    for x, y in zip(xs, ys):
        px = _map(x, xmin, xmax, lm, w - rm)
        py = _map(y, ymin, ymax, h - bm, tm)
        d.ellipse([(px - 2, py - 2), (px + 2, py + 2)], fill="#1f77b4")
    d.text((lm, tm), "EVR1 vs Gradiente STD", fill="#000000")
    d.text((lm, h - bm + 5), "EVR1", fill="#000000")
    d.text((10, tm + 5), "Grad STD", fill="#000000")
    if out_path:
        img.save(out_path)
    return out_path

def plot_hist(rows: List[Dict], key: str = "proj_std", bins: int = 50, out_path: Optional[str] = None, show: bool = False):
    if not rows:
        return None
    w, h = 800, 600
    lm, rm, tm, bm = 60, 20, 20, 40
    img = _create_canvas(w, h)
    d = ImageDraw.Draw(img)
    _draw_axes(d, w, h, lm, rm, tm, bm)
    vals = [float(r.get(key, 0.0)) for r in rows]
    if not vals:
        if out_path:
            img.save(out_path)
        return out_path
    vmin, vmax = min(vals), max(vals)
    if vmin == vmax:
        vmax = vmin + 1e-6
    bin_edges = [vmin + (vmax - vmin) * i / bins for i in range(bins + 1)]
    counts = [0] * bins
    for v in vals:
        idx = int((v - vmin) / (vmax - vmin) * bins)
        if idx == bins:
            idx -= 1
        counts[idx] += 1
    cmax = max(counts) if counts else 1
    bar_w = (w - lm - rm) / bins
    for i, c in enumerate(counts):
        x0 = lm + i * bar_w
        x1 = x0 + bar_w - 1
        y1 = h - bm
        y0 = y1 - (c / cmax) * (h - tm - bm)
        d.rectangle([(x0, y0), (x1, y1)], fill="#1f77b4")
    d.text((lm, tm), f"Histograma de {key}", fill="#000000")
    d.text((lm, h - bm + 5), key, fill="#000000")
    d.text((10, tm + 5), "Frecuencia", fill="#000000")
    if out_path:
        img.save(out_path)
    return out_path
