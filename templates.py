"""
Glyph template bank — heuristic fallback reader.
Renders charset x fonts x sizes x rotations, matches by blurred cosine sim.
"""
import functools
import string

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

GLYPH_BOX = 28
# Use THE SAME FONTS the generator draws with (generator.FONT_FILES): the
# template bank then matches the captive font set exactly.  System-font
# fallbacks remain for machines where none of them exist.
try:
    import generator as _gen
    FONTS = [p for p in _gen.FONT_FILES]
except Exception:
    FONTS = []
FONTS += [
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
]
SIZES = (26, 30, 34, 38, 42)
ROTS = (-10, -5, 0, 5, 10)
CHARSET = string.ascii_letters + string.digits


def _square_norm(crop: np.ndarray) -> np.ndarray:
    """pad bbox to square (preserves aspect), resize to GLYPH_BOX"""
    h, w = crop.shape
    s = max(h, w)
    canvas = np.zeros((s, s), crop.dtype)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    q = Image.fromarray(canvas).resize((GLYPH_BOX, GLYPH_BOX), Image.LANCZOS)
    return (np.array(q) > 100).astype(np.float32)


def _soft(bin28: np.ndarray) -> np.ndarray:
    f = cv2.GaussianBlur(bin28.astype(np.float32), (0, 0), 1.3).ravel()
    f = f - f.mean()
    nrm = np.linalg.norm(f)
    return f / nrm if nrm > 1e-6 else f


def _render_norm(ch, font_path, size, rot):
    try:
        font = ImageFont.truetype(font_path, size)
    except OSError:
        return None
    pad = size * 3
    im = Image.new("L", (pad, pad), 0)
    ImageDraw.Draw(im).text((pad // 2, pad // 2), ch, font=font, fill=255,
                            anchor="mm")
    if rot:
        im = im.rotate(rot, resample=Image.BICUBIC, expand=True)
    a = np.array(im)
    ys, xs = np.where(a > 40)
    if len(xs) == 0:
        return None
    return _square_norm(a[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


@functools.lru_cache(maxsize=1)
def bank():
    rows, labels = [], []
    for fp in FONTS:
        for size in SIZES:
            for rot in ROTS:
                for ch in CHARSET:
                    t = _render_norm(ch, fp, size, rot)
                    if t is not None:
                        rows.append(_soft(t))
                        labels.append(ch)
    return np.array(rows, dtype=np.float32), labels


def score_candidates(glyph: np.ndarray, cands) -> list:
    """correlation scores for a few candidate chars (verifier use)."""
    B, labels = bank()
    v = _soft(_square_norm((glyph * 255).astype(np.uint8)))
    out = []
    for ch in cands:
        idx = [i for i, l in enumerate(labels) if l == ch]
        if not idx:
            out.append(-1.0)
            continue
        out.append(float(np.max(B[idx] @ v)))
    return out


def classify(glyph: np.ndarray) -> str:
    v = _soft(_square_norm((glyph * 255).astype(np.uint8)))
    B, labels = bank()
    return labels[int(np.argmax(B @ v))]


def classify_top(glyph: np.ndarray, k: int = 3):
    v = _soft(_square_norm((glyph * 255).astype(np.uint8)))
    B, labels = bank()
    s = B @ v
    order = np.argsort(-s)[:k]
    return [(labels[i], float(s[i])) for i in order]
