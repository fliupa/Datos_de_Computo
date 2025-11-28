import numpy as np
from PIL import Image

def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img)

def luminance(rgb):
    arr = rgb.astype(np.float32) / 255.0
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]

def gradients(luma):
    gy, gx = np.gradient(luma.astype(np.float32))
    return gx, gy

def pca_features(gx, gy):
    x = np.stack([gx.flatten(), gy.flatten()], axis=1)
    x_mean = x.mean(axis=0)
    x_centered = x - x_mean
    cov = np.cov(x_centered, rowvar=False)
    w, v = np.linalg.eig(cov)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]
    evr1 = float(w[0] / (w[0] + w[1]))
    evr2 = float(w[1] / (w[0] + w[1]))
    pc1 = v[:, 0]
    proj = x_centered @ pc1
    proj_mean = float(proj.mean())
    proj_std = float(proj.std())
    if proj_std > 0:
        proj_skew = float(((proj - proj_mean) ** 3).mean() / (proj_std ** 3))
    else:
        proj_skew = 0.0
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mean = float(grad_mag.mean())
    grad_std = float(grad_mag.std())
    return {
        "evr1": evr1,
        "evr2": evr2,
        "proj_mean": proj_mean,
        "proj_std": proj_std,
        "proj_skew": proj_skew,
        "grad_mean": grad_mean,
        "grad_std": grad_std,
    }

def analyze_image(path):
    rgb = load_image(path)
    luma = luminance(rgb)
    gx, gy = gradients(luma)
    feats = pca_features(gx, gy)
    h, w = luma.shape
    feats.update({"width": int(w), "height": int(h), "path": str(path)})
    return feats

def _to_uint8(arr):
    a = arr.astype(np.float32)
    amin = float(a.min())
    amax = float(a.max())
    if amax == amin:
        a = np.zeros_like(a)
    else:
        a = (a - amin) / (amax - amin)
    a = (a * 255.0).clip(0, 255).astype(np.uint8)
    return a

def _save_gray(arr, path):
    img = Image.fromarray(_to_uint8(arr), mode="L")
    img.save(path)

def save_gradient_maps(path, out_dir=None, prefix=None):
    rgb = load_image(path)
    luma = luminance(rgb)
    gx, gy = gradients(luma)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx)
    ang_norm = (ang + np.pi) / (2.0 * np.pi)
    from pathlib import Path
    p = Path(path)
    stem = prefix or p.stem
    d = Path(out_dir) if out_dir else p.parent
    d.mkdir(parents=True, exist_ok=True)
    fp_luma = str(d / f"{stem}_luma.png")
    fp_gx = str(d / f"{stem}_gx.png")
    fp_gy = str(d / f"{stem}_gy.png")
    fp_mag = str(d / f"{stem}_grad_mag.png")
    fp_ang = str(d / f"{stem}_grad_angle.png")
    _save_gray(luma, fp_luma)
    _save_gray(gx, fp_gx)
    _save_gray(gy, fp_gy)
    _save_gray(mag, fp_mag)
    _save_gray(ang_norm, fp_ang)
    return {
        "luma": fp_luma,
        "gx": fp_gx,
        "gy": fp_gy,
        "mag": fp_mag,
        "angle": fp_ang,
    }

def compute_gradient_maps(path):
    rgb = load_image(path)
    luma = luminance(rgb)
    gx, gy = gradients(luma)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx)
    ang_norm = (ang + np.pi) / (2.0 * np.pi)
    return {
        "luma": luma,
        "gx": gx,
        "gy": gy,
        "mag": mag,
        "angle": ang_norm,
    }
