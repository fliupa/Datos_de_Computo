import argparse
import csv
import os
from pathlib import Path
from .pipeline import analyze_image, save_gradient_maps
from .visualize import plot_scatter, plot_hist

def find_images(root, recursive):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    p = Path(root)
    if p.is_file():
        return [p] if p.suffix.lower() in exts else []
    if recursive:
        return [q for q in p.rglob("*") if q.suffix.lower() in exts]
    return [q for q in p.glob("*") if q.suffix.lower() in exts]

def run(input_path, output_csv, recursive, do_plot, plot_file, show, grad_maps, grad_dir, output_dir, name_by_image):
    files = find_images(input_path, recursive)
    rows = []
    # Resolve output CSV path
    out_csv = output_csv
    if name_by_image and len(files) == 1:
        from pathlib import Path
        p = files[0]
        base_dir = Path(output_dir) if output_dir else p.parent
        base_dir.mkdir(parents=True, exist_ok=True)
        out_csv = str(base_dir / f"{p.stem}_resultados.csv")
    elif output_dir and not output_csv:
        # If an output directory is provided without a CSV filename
        # and a single file is given, auto-name by image
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if len(files) == 1:
            p = files[0]
            out_csv = str(Path(output_dir) / f"{p.stem}_resultados.csv")
    for f in files:
        try:
            rows.append(analyze_image(f))
        except Exception as e:
            pass
    headers = [
        "path",
        "width",
        "height",
        "evr1",
        "evr2",
        "proj_mean",
        "proj_std",
        "proj_skew",
        "grad_mean",
        "grad_std",
    ]
    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in headers})
    else:
        print(",".join(headers))
        for r in rows:
            print(",".join(str(r.get(k, "")) for k in headers))
    if do_plot and rows:
        default_png = None
        if out_csv and not plot_file:
            base = os.path.splitext(out_csv)[0]
            default_png = base + "_scatter.png"
            default_hist = base + "_hist.png"
        scatter_out = plot_file or default_png
        hist_out = None
        if out_csv and not plot_file:
            hist_out = default_hist
        plot_scatter(rows, out_path=scatter_out, show=show)
        plot_hist(rows, key="proj_std", out_path=hist_out, show=show)
    if grad_maps and files:
        target_dir = grad_dir
        if not target_dir and out_csv:
            target_dir = os.path.dirname(out_csv) or None
        for f in files:
            try:
                save_gradient_maps(str(f), out_dir=target_dir)
            except Exception as e:
                pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot-file", default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--grad-maps", action="store_true")
    ap.add_argument("--grad-dir", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--name-by-image", action="store_true")
    args = ap.parse_args()
    run(args.input, args.output, args.recursive, args.plot, args.plot_file, args.show, args.grad_maps, args.grad_dir, args.output_dir, args.name_by_image)

if __name__ == "__main__":
    main()
