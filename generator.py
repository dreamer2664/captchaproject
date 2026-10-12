"""
Captcha generator v3 — calibrated on the REAL page screenshot
(uploads/immagine_2026-10-09_211228957.png, 2026-10-09).

What the real thing does (measured from the screenshot + user reports):
  * card ~172x70 native, but the IMAGE SIZE CHANGES every round
  * pale card backgrounds (khaki (236,224,167) in the shot) that change
  * glyph ink dark & saturated (navy/purple (82,64,197) in the shot),
    DIFFERENT COLOR PER CHARACTER, different FONT per character,
    different SIZE per character, rotated
  * noise: 1px scribble lines in random mid hues at partial alpha,
    plus scattered 1px speckle dots over the whole card
  * dual mode: up to two cards at once

generate()            -> (img, text)
generate(boxes=True)  -> (img, text, [(xc, baseline, em), ...])
"""
import math
import random
import string

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 172, 70
CHARS = string.ascii_letters + string.digits

# --- palettes measured / extrapolated from the real page -------------------
CARD_BGS = [
    (236, 224, 167), (243, 239, 158),          # the khakis from the screenshot
    (214, 228, 240), (240, 219, 222),          # pale blue / pink rounds
    (219, 235, 209), (248, 248, 246),          # pale green / near-white
    (228, 222, 240), (250, 229, 197),          # pale lavender / peach
    (222, 236, 236), (240, 240, 210),
]
INK_COLORS = [
    (40, 40, 150), (82, 64, 197), (30, 30, 30), (140, 30, 35),
    (25, 105, 45), (170, 80, 20), (110, 30, 140), (0, 105, 120),
    (60, 60, 60), (35, 90, 160),
]
NOISE_COLORS = [
    (60, 180, 90), (220, 110, 85), (110, 107, 157), (230, 180, 60),
    (90, 200, 200), (200, 90, 180), (150, 200, 120), (210, 140, 100),
]
_FONT_CANDIDATES = [
    # system families (paths differ per distro / sandbox generation)
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    # matplotlib's bundled faces (always present wherever mpl is)
    "/usr/local/lib/python3.13/site-packages/matplotlib/mpl-data/fonts/ttf/STIXGeneral.ttf",
    "/usr/local/lib/python3.13/site-packages/matplotlib/mpl-data/fonts/ttf/STIXNonUni.ttf",
    "/usr/local/lib/python3.13/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSerifDisplay.ttf",
    "C:/Windows/Fonts/times.ttf",      # user's Windows box, if training there
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/georgia.ttf",
]
import os as _os


def _font_has_glyphs(path: str) -> bool:
    """some faces (STIXNonUni, DejaVuSerifDisplay) silently render DIGITS as
    nothing — a captcha font pool must prove it can draw 0-9 and A-z"""
    try:
        from PIL import Image as _I, ImageDraw as _D
        im = _I.new("RGBA", (200, 60), (0, 0, 0, 0))
        _D.Draw(im).text((10, 10), "0aZ9", font=_I_font_truetype(path, 30),
                         fill=(0, 0, 0, 255))
        return im.getbbox() is not None
    except Exception:
        return False


def _I_font_truetype(path, size):
    from PIL import ImageFont as _IF
    return _IF.truetype(path, size)


FONT_FILES = [p for p in _FONT_CANDIDATES
              if _os.path.exists(p) and _font_has_glyphs(p)] or \
             [_FONT_CANDIDATES[4]]
_font_cache: dict = {}


def _font(path: str, size: int):
    key = (path, size)
    f = _font_cache.get(key)
    if f is None:
        f = ImageFont.truetype(path, size)
        _font_cache[key] = f
    return f


# --- hard-mode background designs: objects & patterns behind the glyphs ----
DESIGN_COLORS = [
    (180, 120, 90), (90, 140, 180), (140, 180, 90), (180, 90, 140),
    (120, 120, 190), (190, 160, 90), (100, 170, 160), (170, 130, 170),
]


def _paint_design(d, W, H):
    """objects + patterns behind the text, like hostile real captchas"""
    kind = random.randint(0, 5)
    cols = DESIGN_COLORS
    if kind == 0:                                  # stripes
        c = random.choice(cols); a = random.randint(25, 50)
        step = random.randint(6, 16)
        for i in range(0, W + H, step):
            d.line([(i, 0), (i - H, H)], fill=c + (a,), width=random.randint(2, 4))
    elif kind == 1:                                # checkerboard
        c = random.choice(cols); a = random.randint(22, 45)
        s = random.randint(8, 20)
        for yy in range(0, H, s):
            for xx in range(0, W, s):
                if (xx // s + yy // s) % 2 == 0:
                    d.rectangle([xx, yy, xx + s - 1, yy + s - 1], fill=c + (a,))
    elif kind == 2:                                # big soft blobs (objects)
        for _ in range(random.randint(2, 4)):
            c = random.choice(cols); a = random.randint(25, 50)
            x0, y0 = random.randint(-20, W), random.randint(-10, H)
            r = random.randint(15, 45)
            d.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=c + (a,))
    elif kind == 3:                                # rings / outlines
        for _ in range(random.randint(2, 4)):
            c = random.choice(cols); a = random.randint(50, 90)
            x0, y0 = random.randint(0, W), random.randint(0, H)
            r = random.randint(10, 35)
            d.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], outline=c + (a,),
                      width=random.randint(2, 4))
    elif kind == 4:                                # star / polygon objects
        import math
        for _ in range(random.randint(1, 3)):
            c = random.choice(cols); a = random.randint(40, 80)
            cx, cy = random.randint(10, W - 10), random.randint(10, H - 10)
            r = random.randint(12, 30); pts = []
            k = random.choice([5, 6, 3])
            for i in range(k * 2):
                rr = r if i % 2 == 0 else r * 0.45
                ang = math.pi * i / k + random.uniform(-0.2, 0.2)
                pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
            d.polygon(pts, fill=c + (a,))
    else:                                          # radial gradient wash
        c = random.choice(cols)
        cx, cy = random.randint(0, W), random.randint(0, H)
        for r in range(70, 0, -6):
            a = max(6, int(40 * (1 - r / 70)))
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c + (a,))


LAST_BOXES: list = []


def generate(length: int | None = None, boxes: bool = False,
             style: str = "real"):
    """One captcha card.
    style="easy": clean binary (white bg/black ink or black bg/white ink),
                  single font, no noise — the 92% bar suite
    style="hard": designed backgrounds (objects/patterns), per-char font /
                  color / size, 4-8 chars, heavy noise — the 85% bar suite
    style="real": replica of the user's real page cards (default)
    """
    if style == "easy":
        return _generate_easy(length or 6, boxes)
    n = length or (random.randint(4, 8) if style == "hard" else 6)
    hard = style == "hard"
    W = min(340, max(140, 24 * n + 28))   # real-page density (~24 px/char)
    bg = random.choice(CARD_BGS)
    card = Image.new("RGBA", (W, HEIGHT), bg + (255,))
    if hard:
        des = Image.new("RGBA", (W, HEIGHT), (0, 0, 0, 0))
        _paint_design(ImageDraw.Draw(des, "RGBA"), W, HEIGHT)
        card = Image.alpha_composite(card, des)   # real blending (ImageDraw
    layer = Image.new("RGBA", (W, HEIGHT), (0, 0, 0, 0))   # writes alpha raw)

    # per-character: font, size, color, rotation  (the hard-captcha axes)
    adv_total = 0
    specs = []
    for _ in range(n):
        size = random.randint(26, 44)
        specs.append(dict(
            ch=random.choice(CHARS),
            font=random.choice(FONT_FILES),
            size=size,
            color=random.choice(INK_COLORS),
            rot=random.uniform(-12, 12),
        ))
    # lay out with small random gaps, then center the run; shrink until it
    # fits with a margin like the real card (never clips at the edges)
    widths = []
    for s in specs:
        widths.append(_font(s["font"], s["size"]).getlength(s["ch"]))
    gaps = [random.randint(0, 4) for _ in range(n)]
    for _shrink in range(6):
        run = sum(widths) + sum(gaps)
        if run <= W - 14:
            break
        for i, s in enumerate(specs):
            s["size"] = max(18, int(s["size"] * 0.88))
            widths[i] = _font(s["font"], s["size"]).getlength(s["ch"])
    run = sum(widths) + sum(gaps)
    slack = max(0.0, (W - 14 - run) / 2)
    x = 7 + slack / 2 + random.uniform(-min(4, slack / 2), min(4, slack / 2))

    char_boxes = []
    text = ""
    for i, s in enumerate(specs):
        f = _font(s["font"], s["size"])
        pad = s["size"] * 2
        glyph = Image.new("RGBA", (pad, pad), (0, 0, 0, 0))
        ImageDraw.Draw(glyph).text((pad // 2, pad // 2), s["ch"], font=f,
                                   fill=s["color"] + (255,), anchor="mm")
        glyph = glyph.rotate(s["rot"], resample=Image.BICUBIC, expand=True)
        yc = HEIGHT // 2 + random.randint(-6, 6)
        layer.alpha_composite(glyph, (int(x + widths[i] / 2 - glyph.width / 2),
                                      int(yc - glyph.height / 2)))
        asc, _desc = f.getmetrics()
        char_boxes.append((x + widths[i] / 2, yc - s["size"] / 2 + asc,
                           s["size"]))
        text += s["ch"]
        x += widths[i] + gaps[i]

    card = Image.alpha_composite(card, layer)

    # --- noise ON TOP, exactly like the shot: thin scribbles + speckle -----
    over = Image.new("RGBA", (W, HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(over, "RGBA")
    for _ in range(random.randint(4, 7) if hard else random.randint(3, 5)):
        c = random.choice(NOISE_COLORS)
        a = random.randint(70, 125)
        x0, y0 = random.randint(-10, W // 2), random.randint(0, HEIGHT)
        x1, y1 = random.randint(W // 2, W + 10), random.randint(0, HEIGHT)
        d.line([(x0, y0), (x1, y1)], fill=c + (a,), width=1)
    for _ in range(random.randint(40, 90) if hard else random.randint(25, 55)):
        c = random.choice(NOISE_COLORS + INK_COLORS)
        a = random.randint(90, 200)
        px, py = random.randint(0, W - 1), random.randint(0, HEIGHT - 1)
        d.point((px, py), fill=c + (a,))

    card = Image.alpha_composite(card, over)
    img = card.convert("RGB")
    LAST_BOXES.clear()
    LAST_BOXES.extend(char_boxes)
    if boxes:
        return img, text, list(char_boxes)
    return img, text


def _generate_easy(n: int, boxes: bool):
    """clean binary captcha: white bg + black ink, or black bg + white ink"""
    invert = random.random() < 0.5
    bg = (0, 0, 0) if invert else (255, 255, 255)
    ink = (255, 255, 255) if invert else (0, 0, 0)
    W = min(300, max(150, 30 * n + 30))
    card = Image.new("RGBA", (W, HEIGHT), bg + (255,))
    layer = Image.new("RGBA", (W, HEIGHT), (0, 0, 0, 0))
    font_path = random.choice(FONT_FILES)
    card_size = random.randint(30, 40)     # ONE size per card: cap/x-height
    specs = []                             # lines stay exact for case reads
    for _ in range(n):
        specs.append(dict(ch=random.choice(CHARS), font=font_path,
                          size=card_size, color=ink,
                          rot=random.uniform(-10, 10)))
    widths = [_font(s["font"], s["size"]).getlength(s["ch"]) for s in specs]
    gaps = [random.randint(2, 6) for _ in range(n)]
    for _shrink in range(6):
        run = sum(widths) + sum(gaps)
        if run <= W - 14:
            break
        for i, s in enumerate(specs):
            s["size"] = max(20, int(s["size"] * 0.9))
            widths[i] = _font(s["font"], s["size"]).getlength(s["ch"])
    run = sum(widths) + sum(gaps)
    x = (W - run) / 2
    char_boxes = []
    text = ""
    for i, s in enumerate(specs):
        f = _font(s["font"], s["size"])
        pad = s["size"] * 2
        glyph = Image.new("RGBA", (pad, pad), (0, 0, 0, 0))
        ImageDraw.Draw(glyph).text((pad // 2, pad // 2), s["ch"], font=f,
                                   fill=s["color"] + (255,), anchor="mm")
        glyph = glyph.rotate(s["rot"], resample=Image.BICUBIC, expand=True)
        yc = HEIGHT // 2 + random.randint(-3, 3)
        layer.alpha_composite(glyph, (int(x + widths[i] / 2 - glyph.width / 2),
                                      int(yc - glyph.height / 2)))
        asc, _desc = f.getmetrics()
        char_boxes.append((x + widths[i] / 2, yc - s["size"] / 2 + asc,
                           s["size"]))
        text += s["ch"]
        x += widths[i] + gaps[i]
    img = Image.alpha_composite(card, layer).convert("RGB")
    LAST_BOXES.clear()
    LAST_BOXES.extend(char_boxes)
    if boxes:
        return img, text, list(char_boxes)
    return img, text
