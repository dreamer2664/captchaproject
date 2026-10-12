"""
The "reads and writes" model — solver chain.

    solve(image, expected_len) -> str     case-sensitive best-effort read

Chain: glyphnet (trained CNN, batched) -> gemini vision (keys + low
confidence + HARD deadline) -> glyph templates -> tesseract -> rapidocr.

Segmentation is COLOR-INVARIANT and LENGTH-AWARE: background = median
color, foreground = pixels far from it, component repairs (tittle merge,
valley split, gap merge, sliver merge), then — when more components than
expected characters survive (noise lines, page junk) — the most glyph-like
subset is selected by height band + area.
"""
import re
import string
import time

import cv2
import numpy as np
from PIL import Image

try:
    import pytesseract
except Exception:                     # OCR fallback is optional; the CNN is not
    pytesseract = None

CHARSET = string.ascii_letters + string.digits


# ------------------------------------------------------------ segmentation
def _bg_median(a: np.ndarray) -> np.ndarray:
    """median background color, subsampled — fast even on huge captures"""
    flat = a.reshape(-1, 3)
    if len(flat) > 160_000:
        flat = flat[:: len(flat) // 160_000]
    return np.median(flat, axis=0)


def analyze(img: Image.Image, diff_thresh: int = 62, min_area: int = 12):
    """SINGLE pass -> (dom, comps).
    dom:   signed contrast field in [0,1]; glyphs >= 0.875, lines <= 0.75
           (identical to crnn.contrast_field on normal-size images)
    comps: repaired glyph components (x,y,w,h,m,area), left-to-right.
    diff_thresh 62: generator lines cap at dist 60 => invisible to the mask;
    glyph ink (lum contrast > 70) always clears it."""
    a = np.array(img.convert("RGB")).astype(np.float32)
    bg = _bg_median(a)
    lum = a.mean(axis=2)
    bg_lum = float(bg.mean())
    diff = np.abs(a - bg).max(axis=2)
    mask = (diff > diff_thresh).astype(np.uint8)
    if mask.size <= 250_000:
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        sel = opened if opened.sum() > 20 else mask
    else:
        sel = mask
    # frame-ring guard: if the crop's perimeter is mostly "ink", the capture
    # contains a sliver/ring of page background around the card (loose box) —
    # that frame would fuse every glyph into full-height strips.  Strip it.
    H, W = mask.shape
    if H > 12 and W > 12:
        ring = np.concatenate([mask[:2, :].ravel(), mask[-2:, :].ravel(),
                               mask[:, :2].ravel(), mask[:, -2:].ravel()])
        if ring.mean() > 0.45:
            mask[:3, :] = 0; mask[-3:, :] = 0
            mask[:, :3] = 0; mask[:, -3:] = 0
    # sign AFTER the ring strip: a sliver of bright page background around a
    # loose crop would otherwise flip the polarity and kill the whole dom
    sign = 1.0
    if mask.sum() > 20:
        inside_lum = float(np.median(lum[mask > 0]))
        sign = 1.0 if inside_lum < bg_lum else -1.0
    dom = np.clip(sign * (bg_lum - lum), 0, 80) / 80.0
    if H > 12 and W > 12 and ring.mean() > 0.45:
        dom[:3, :] = 0; dom[-3:, :] = 0
        dom[:, :3] = 0; dom[:, -3:] = 0
    # SOLID-INK MASK: glyph ink is opaque (dom ~1.0); background objects,
    # patterns and scribbles are semi-transparent (dom <= ~0.6).  Building the
    # components from dom instead of colour-diff keeps objects from ever
    # touching (and fusing with) the glyphs.
    ink = ((dom >= 0.88) & (mask > 0)).astype(np.uint8)
    if ink.sum() < 20:                     # faint capture: fall back to mask
        ink = mask
    comps = _components(ink, min_area)
    return dom, comps


def _components(mask: np.ndarray, min_area: int = 20):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    H, W = mask.shape
    comps = []
    for i in range(1, n):
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        if (h >= H - 3 and w <= 0.18 * W) or (w >= W - 3 and h <= 0.18 * H):
            continue                          # frame sliver: full-span strip
        comps.append({"x": x, "y": y, "w": w, "h": h,
                      "m": (labels[y:y + h, x:x + w] == i).astype(np.uint8),
                      "area": int(stats[i, cv2.CC_STAT_AREA])})
    if len(comps) > 300:                    # pathological capture: cap early
        comps.sort(key=lambda c: -c["area"])
        comps = comps[:300]

    # 1) re-attach tittles / fragments sitting right above a bigger comp
    comps.sort(key=lambda c: -c["area"])
    kept = []
    for c in comps:
        merged = False
        if c["area"] < min_area:
            for k in kept:
                xov = min(c["x"] + c["w"], k["x"] + k["w"]) - max(c["x"], k["x"])
                gap = k["y"] - (c["y"] + c["h"])
                if xov > 0 and 0 <= gap <= 10:
                    _merge_into(k, c)
                    merged = True
                    break
        if not merged and c["area"] >= min_area // 2:
            kept.append(c)
    kept = [c for c in kept if c["area"] >= min_area]

    # 2) split touching pairs at projection valleys
    singles = [c["w"] for c in kept if c["w"] <= 1.5 * c["h"]]
    med_w = sorted(singles)[len(singles) // 2] if singles else 20
    out = []
    for c in kept:
        out.extend(_split_wide(c, med_w))

    # 3) re-attach fragments: gap<=1 px AND the smaller side is a fragment
    # (tittle, noise-cut sliver).  Two FULL glyphs 1px apart must stay split:
    # merged they read as one character and the length dies.
    med_a = sorted(c["area"] for c in out)[len(out) // 2] if out else 20
    out.sort(key=lambda c: c["x"])
    changed = True
    while changed:
        changed = False
        for i in range(len(out) - 1):
            a, b = out[i], out[i + 1]
            gap = b["x"] - (a["x"] + a["w"])
            union_w = a["w"] + b["w"] + gap
            if (gap <= 1 and union_w <= 1.6 * med_w
                    and (min(a["area"], b["area"]) < 0.45 * med_a
                         or union_w <= 1.15 * med_w)):
                _merge_into(a, b)
                out.pop(i + 1)
                changed = True
                break
    # 4) re-attach sliver fragments (broken stroke junctions: v w x y 4 ...)
    out.sort(key=lambda c: c["x"])
    changed = True
    while changed:
        changed = False
        for i, c in enumerate(out):
            if c["w"] >= 0.5 * med_w:
                continue
            for j in (i - 1, i + 1):
                if not (0 <= j < len(out)):
                    continue
                o = out[j]
                gap = max(o["x"] - (c["x"] + c["w"]), c["x"] - (o["x"] + o["w"]))
                vov = (min(c["y"] + c["h"], o["y"] + o["h"]) - max(c["y"], o["y"]))
                if gap <= 3 and vov > 0.4 * min(c["h"], o["h"]) \
                        and c["w"] + o["w"] + gap <= 1.35 * med_w:
                    _merge_into(o, c)
                    out.pop(i)
                    changed = True
                    break
            if changed:
                break
    out.sort(key=lambda c: c["x"])
    return out


def select_glyphs(comps, expected_len=None):
    """-> (kept, dropped).  When MORE components than expected characters
    survive (strong noise lines, page junk around the captcha), keep the
    most glyph-like subset: height band around the text height, then the
    largest areas, then left-to-right order."""
    if not expected_len or len(comps) <= expected_len:
        return comps, []
    by_area = sorted(comps, key=lambda c: -c["area"])
    th = float(np.median([c["h"] for c in by_area[: expected_len * 2]]))
    band = [c for c in comps if 0.55 * th <= c["h"] <= 1.8 * th]
    if len(band) > expected_len:
        band = sorted(band, key=lambda c: -c["area"])[: expected_len]
    if not band:
        band = by_area[: expected_len]
    kept = sorted(band, key=lambda c: c["x"])
    kset = {id(c) for c in kept}
    return kept, [c for c in comps if id(c) not in kset]


def glyph_components(img: Image.Image, diff_thresh: int = 62, min_area: int = 20):
    """-> list of glyph component dicts (x,y,w,h,m,area), left-to-right."""
    return analyze(img, diff_thresh, min_area)[1]


def _merge_into(k: dict, c: dict) -> None:
    x0, y0 = min(k["x"], c["x"]), min(k["y"], c["y"])
    x1, y1 = max(k["x"] + k["w"], c["x"] + c["w"]), max(k["y"] + k["h"], c["y"] + c["h"])
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    m[k["y"] - y0:k["y"] - y0 + k["h"], k["x"] - x0:k["x"] - x0 + k["w"]] |= k["m"]
    m[c["y"] - y0:c["y"] - y0 + c["h"], c["x"] - x0:c["x"] - x0 + c["w"]] |= c["m"]
    k.update(x=x0, y=y0, w=x1 - x0, h=y1 - y0, m=m, area=k["area"] + c["area"])


def _split_wide(c: dict, med_w: float, max_depth: int = 2) -> list[dict]:
    # width gate vs the comp's OWN height: mixed-size captchas make a global
    # median width meaningless (a 44px 'M' is 2x a 26px-glyph median and was
    # being cut in half).  Single glyphs are w <= ~1.0h; touching pairs >= 1.4h
    if c["w"] <= 1.40 * c["h"] or max_depth == 0:
        return [c]
    col = c["m"].sum(axis=0)
    lo, hi = int(c["w"] * 0.25), int(c["w"] * 0.75)
    if hi <= lo:
        return [c]
    cut = lo + int(np.argmin(col[lo:hi]))
    peak = float(col.max())
    # touching PAIRS (w >= 1.9 med) may be cut at a shallow valley (ink of both
    # glyphs grazes the cut column); single wide glyphs (m w M W ~1.5-1.8 med)
    # never reach the width gate, so their stems stay intact
    if col[cut] > 0.60 * peak:
        return [c]
    parts = []
    for m, ox in ((c["m"][:, :cut], 0), (c["m"][:, cut:], cut)):
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        mm = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        parts.append({"x": c["x"] + ox + xs.min(), "y": c["y"] + ys.min(),
                      "w": mm.shape[1], "h": mm.shape[0], "m": mm,
                      "area": int(mm.sum())})
    if len(parts) == 2:
        w0, w1 = parts[0]["w"], parts[1]["w"]
        if min(w0, w1) < 0.42 * c["h"]:
            return [c]                  # sliver cut: probably one wide glyph
    res = []
    for p in parts:
        res.extend(_split_wide(p, med_w, max_depth - 1))
    return res or [c]


def glyph_mask(img: Image.Image, **kw):
    return [c["m"] for c in glyph_components(img, **kw)]


def preprocess(img: Image.Image, min_area: int = 20) -> np.ndarray:
    """clean black-on-white binary for the generic OCR engines"""
    a = np.array(img.convert("RGB")).astype(np.float32)
    bg = _bg_median(a)
    m = (np.abs(a - bg).max(axis=2) > 62).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    out = cv2.resize(out, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, out = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)
    return 255 - out


def _clean(text: str, expected_len: int | None = None) -> str:
    t = re.sub(r"[^A-Za-z0-9]", "", text)          # keep case!
    if expected_len and len(t) > expected_len:
        t = t[:expected_len]
    return t


TARGET_H = 70          # native captcha height the CNN was trained on


def norm_scale(img: Image.Image) -> Image.Image:
    """Bring a capture to the ~70px height the CNN was trained on, but ONLY
    when it is genuinely far off.  A live screen grab is native pixels (1x),
    so this is a no-op in the real runner; it exists for hi-dpi/zoomed
    captures.  Near-native crops must pass through untouched: LANCZOS
    resampling shifts glyph ink off the subpixel grid the model learned and
    costs measurable accuracy."""
    W, H = img.size
    if H < 1:
        return img
    scale = TARGET_H / float(H)
    if 0.80 <= scale <= 1.25:          # already ~native: do not resample
        return img
    return img.resize((max(1, int(round(W * scale))), TARGET_H), Image.LANCZOS)


# ------------------------------------------------------------ readers
def read_tesseract(img: Image.Image, expected_len: int | None = None) -> str:
    bin_img = preprocess(img)
    cfg = f"--psm 7 -c tessedit_char_whitelist={CHARSET}"
    if pytesseract is None:
        return ""
    t = _clean(pytesseract.image_to_string(bin_img, config=cfg), expected_len)
    if expected_len and len(t) != expected_len:
        raw = pytesseract.image_to_string(
            bin_img, config=f"--psm 8 -c tessedit_char_whitelist={CHARSET}")
        t = _clean(raw, expected_len) or t
    return t


def read_rapidocr(img: Image.Image, expected_len: int | None = None) -> str:
    from rapidocr_onnxruntime import RapidOCR
    if not hasattr(read_rapidocr, "_r"):
        read_rapidocr._r = RapidOCR()
    bin_img = preprocess(img)
    result, _ = read_rapidocr._r(bin_img)
    if not result:
        result, _ = read_rapidocr._r(np.array(img.convert("RGB")))
    text = "".join(line[1] for line in result) if result else ""
    return _clean(text, expected_len)


def upright(m: np.ndarray) -> np.ndarray:
    """de-rotate a glyph mask: angle minimizing bounding-box area"""
    best, best_area = m, m.shape[0] * m.shape[1]
    h, w = m.shape
    cx, cy = w / 2, h / 2
    for a in range(-20, 21, 2):
        if a == 0:
            r = m
        else:
            M = cv2.getRotationMatrix2D((cx, cy), a, 1.0)
            r = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST)
        ys, xs = np.where(r)
        if len(xs) == 0:
            continue
        area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
        if area < best_area:
            best_area = area
            best = r[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return best


TOP_CAP = set(string.ascii_uppercase + string.digits + "bdfhklt")
BOT_DESC = set("gjpqy")


def read_template(img: Image.Image, expected_len: int | None = None) -> str:
    """segment, de-rotate, template-match, rescore by line placement"""
    import templates
    comps = glyph_components(norm_scale(img))
    comps, _ = select_glyphs(comps, expected_len)
    if not comps:
        return ""
    for c in comps:
        u = upright(c["m"])
        c["u"], c["uh"] = u, u.shape[0]
    bots = [c["y"] + c["h"] for c in comps]
    baseline = float(np.median(bots))
    desc_bots = [b for b in bots if b > baseline + 4]
    desc_line = float(np.mean(desc_bots)) if desc_bots else baseline
    tops = [baseline - c["uh"] if b <= baseline + 3 else desc_line - c["uh"]
            for c, b in zip(comps, bots)]
    tops_base = [t for t, b in zip(tops, bots) if b <= baseline + 3]
    cap_line = float(min(tops_base)) if tops_base else float(min(tops))
    xs_t = [t for t in tops_base if t > cap_line + 4]
    cap = max(baseline - cap_line, 8.0)
    x_line = float(np.median(xs_t)) if xs_t else cap_line + 0.28 * cap
    out = []
    for c, top, bot in zip(comps, tops, bots):
        best_ch, best_sc = None, -1e9
        for ch, sc in templates.classify_top(c["u"], k=6):
            et = cap_line if ch in TOP_CAP else x_line
            eb = desc_line if ch in BOT_DESC else baseline
            sc -= 1.0 * (abs(top - et) + abs(bot - eb)) / cap
            if sc > best_sc:
                best_ch, best_sc = ch, sc
        out.append(best_ch)
    return "".join(out)


AMBIG_CASE = set("acemnorsuvwxz")   # x-height shapes: case = pure scale
XHEIGHT_REF = set("acemnr")          # unambiguous x-height-only shapes


def _baseline(comps) -> float:
    """Baseline estimate that survives descenders and AA erosion:
    component bottoms form a main cluster at [B-3, B] (flat-bottom glyphs at
    B, round glyphs eroded 1-3px up) plus optional descender spikes at
    [B+7, B+13].  Cut descenders at the largest downward gap >= 5px, then
    average the LOWER half of the main cluster (flat bottoms = truth)."""
    bots = np.sort(np.array([c["y"] + c["h"] for c in comps], float))
    if len(bots) == 0:
        return 0.0
    if len(bots) > 1:
        gaps = np.diff(bots)
        gi = int(np.argmax(gaps))
        if gaps[gi] >= 5.0 and gi >= len(bots) - 4:   # few glyphs BELOW the gap
            bots = bots[: gi + 1]                     # keep main cluster side
    k = max(1, len(bots) // 2)
    return float(bots[-k:].mean())


def read_glyphnet_debug(img: Image.Image, expected_len: int | None = None):
    """-> (text, mean_confidence, kept_comps, dropped_comps, dom).
    Batched CNN + length-aware selection + per-image case consistency:
    cap/x-height estimated from the unambiguous glyphs, then scale-ambiguous
    predictions get their case flipped to the line their height matches."""
    import glyphnet
    if glyphnet.get_model() is None:
        return "", 0.0, [], [], None
    img = norm_scale(img)
    dom, comps = analyze(img)
    if not comps:
        return "", 0.0, [], [], dom
    kept, dropped = select_glyphs(comps, expected_len)
    baseline = _baseline(kept)
    xcs = []
    for c in kept:
        ys, xs = np.where(c["m"])
        xcs.append(c["x"] + float(xs.mean()))
    res = (glyphnet.classify_many_tta(dom, xcs, baseline)
           if getattr(glyphnet.get_model(), "tta", True)
           else glyphnet.classify_many(dom, xcs, baseline))
    # ENSEMBLE: a second weights file (glyphnet8.pt) votes per glyph; the
    # higher-confidence read wins disagreements (independent inits/errors)
    m2 = glyphnet.get_model("glyphnet8.pt", quiet=True)
    if m2 is not None and res and res[0][0]:
        res2 = glyphnet.classify_many_tta(dom, xcs, baseline, model=m2)
        merged = []
        for (c1, f1), (c2, f2) in zip(res, res2):
            if c1 == c2:
                merged.append((c1, (f1 + f2) / 2))
            else:
                merged.append((c1, f1 * 1.0) if f1 >= f2 else (c2, f2))
        res = merged
    out = [ch for ch, _ in res]
    confs = [cf for _, cf in res]
    # CASE POST-PASS (all models): vertical geometry is exact information —
    # cap line vs x-height line separate 'C' from 'c' with ~10 px of margin.
    # The CNN case head gets a vote, but a glyph sitting ON the x-height line
    # is lowercase no matter what the head thought.
    heights = [c["h"] for c in kept]
    # LINE ESTIMATE by BIMODAL HEIGHT SPLIT: class-medians get polluted by the
    # very misreads we are fixing (one 'S' read for 's' drags the cap median
    # down to x-height and disables the whole pass).  Heights cluster at
    # {cap/asc} and {x-height}; the widest gap >= 4 px separates them.
    hs = sorted(h for ch, h in zip(out, heights)
                if ch.lower() not in set("gjpqy"))
    cap = xh = None
    if len(hs) >= 2:
        gaps = np.diff(hs)
        gi = int(np.argmax(gaps))
        if gaps[gi] >= 4:
            up, lo = hs[gi + 1:], hs[: gi + 1]
            # uniform-size cards only: mixed sizes smear both clusters and
            # the "lines" become meaningless (hard style keeps mixed sizes)
            if (max(up) - min(up) <= 4) and (max(lo) - min(lo) <= 4):
                cap = float(np.median(up))
                xh = float(np.median(lo))
    if cap is not None and xh is not None:
        if cap - xh > 3:
            for i, ch in enumerate(out):
                if ch.lower() in AMBIG_CASE and abs(heights[i] - cap) != abs(heights[i] - xh):
                    want_upper = abs(heights[i] - cap) < abs(heights[i] - xh)
                    out[i] = ch.upper() if want_upper else ch.lower()
    # GEOMETRY PASS 2: dot-gap and descender cues are exact information the
    # CNN head keeps getting wrong (l/i 47 flips, J/j 43, Z/z 38 in mining).
    #   i,j  : tittle = a row-gap inside the bbox;  l : stem continuous
    #   gjpqy: lowercase tail hangs BELOW baseline; capitals stop at it
    for i, c in enumerate(kept):
        rows = c["m"].sum(axis=1)
        top = int(np.argmax(rows > 0))
        zone = rows[top + 2: top + max(3, int(0.45 * c["h"]))]
        gap = bool(zone.size and 0 in zone)
        bot = c["y"] + c["h"]
        ch = out[i]
        if ch in ("i", "j") and not gap:
            out[i] = "j" if bot > baseline + 3 else "l"
            if out[i] == "l" and ch == "j":
                out[i] = "J" if bot <= baseline + 3 else "j"
        elif ch == "l" and gap:
            out[i] = "j" if bot > baseline + 3 else "i"
        # descender = a THIN TAIL in the bottom quarter (stem width), vs the
        # wide bowl/curve a capital ends with.  Baseline-free: rotation moves
        # component bottoms +-4 px and made baseline rules misfire.
        # descender = thin tail AND a hang below the no-tail baseline.
        # J/P/Y have thin hooks but stop AT the line; rotated cards move
        # plain baseline medians +-3 px, base2 (no-tail comps) is stable.
        if out[i] in set("gjpqy") and bot <= baseline + 2:
            out[i] = out[i].upper()
        elif out[i] in set("GJPQY") and bot > baseline + 3:
            out[i] = out[i].lower()
    # O vs 0: the zero is markedly narrower than the letter O in every
    # proportional font; in monospace they are the same width -> skip there
    mono = any(ch in "1lI" and h > 8 and w / h >= 0.5
               for ch, w, h in zip(out, [c["w"] for c in kept], heights))
    for i, c in enumerate(kept):
        if out[i] in ("O", "0") and c["h"] > 10 and not mono:
            out[i] = "0" if c["w"] / c["h"] < 0.85 else "O"
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return "".join(out), mean_conf, kept, dropped, dom


def read_glyphnet_conf(img: Image.Image, expected_len: int | None = None):
    """-> (text, mean_confidence); ('', 0) if untrained"""
    return read_glyphnet_debug(img, expected_len)[:2]


def read_glyphnet(img: Image.Image, expected_len: int | None = None) -> str:
    return read_glyphnet_debug(img, expected_len)[0]


def read_crnn(img: Image.Image, expected_len: int | None = None) -> str:
    import crnn
    return crnn.read_crnn(img, expected_len)


def read_gemini(img: Image.Image, expected_len: int | None = None,
                timeout: float | None = None, deadline: float | None = None,
                max_attempts: int | None = None) -> str:
    import gemini_reader
    kw = {}
    if timeout is not None:
        kw["timeout"] = timeout
    if deadline is not None:
        kw["deadline"] = deadline
    if max_attempts is not None:
        kw["max_attempts"] = max_attempts
    return gemini_reader.read_gemini(img, expected_len, **kw)


GEMINI_CONF_THRESHOLD = 0.62   # mean per-glyph softmax; 0.75 fired on ~80% of rounds


def solve_budget(img: Image.Image, expected_len: int | None = None,
                 time_left: float = 15.0, type_time: float = 1.5):
    """-> (text, source).  The 15 s law, hard-deadline edition:
      - local read first (batched CNN + selection, ~0.1 s)
      - BLANK regions never reach the network
      - wrong-length reads get confidence 0 -> fallbacks engage
      - Gemini only when unsure AND the budget survives its timeout +
        typing + margin; hard deadline; at most 4 key attempts
      - template fallback (~0.1 s) is always affordable"""
    import gemini_reader
    t0 = time.time()
    g, conf, kept, _dropped, _dom = read_glyphnet_debug(img, expected_len)
    if not kept:
        return "", "none"                     # blank/expired region: skip round
    src = "glyphnet"
    if expected_len and len(g) != expected_len:
        conf = 0.0
    if gemini_reader.has_keys() and (not g or conf < GEMINI_CONF_THRESHOLD):
        spent = time.time() - t0
        budget = time_left - spent - type_time - 2.5     # safety margin
        if budget > 2.5:
            # Size the per-call timeout from the number of calls we intend to
            # make, so attempts x per_call fits the budget.  Two agreeing reads
            # beat one lucky read, but never at the cost of the round.
            attempts = 3 if budget > 7.0 else 2
            per_call = max(1.0, min(4.0, (budget - 1.0) / attempts))
            gm = read_gemini(img, expected_len,
                             timeout=per_call,
                             deadline=t0 + time_left - type_time - 2.0,
                             max_attempts=attempts)
            if gm and (expected_len is None or len(gm) == expected_len):
                return gm, "gemini"
            if gm and not g:
                g, src = gm, "gemini"
    if g and (expected_len is None or len(g) == expected_len):
        return g, src
    t = read_template(img, expected_len)
    if t and (expected_len is None or len(t) == expected_len):
        return t, "template"
    return (g or t), src


def solve(img: Image.Image, expected_len: int | None = None) -> str:
    """best-effort case-sensitive read (no time pressure)"""
    return solve_budget(img, expected_len, time_left=1e9)[0]


def char_accuracy(truth: str, guess: str) -> float:
    if not truth:
        return 0.0
    return sum(a == b for a, b in zip(truth, guess)) / len(truth)
