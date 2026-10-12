"""
Per-glyph CNN classifier — the trained "reads" model (cross-entropy).

Input: 2 channels x 56 x 56
  ch0: baseline-aligned ABSOLUTE contrast patch   (case/size cues)
  ch1: square-normalized glyph shape              (identity cues)

Train:  python3 glyphnet.py train [epochs] [samples]
"""
import shutil
import os
import random
import string
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    torch.set_num_threads(2)      # small batches: fewer threads = lower latency
except Exception:
    pass

from crnn import clean_mask, contrast_field
from generator import generate

ALPHABET = string.ascii_letters + string.digits
P = 56
WID = 84          # wide patch: neighbours visible => relative case cues
UP, DOWN, SIDE = 40, 16, 42


def _shape_channel(mask_win: np.ndarray) -> np.ndarray:
    """largest component of window, square-normalized to 28, centred"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask_win.astype(np.uint8), 8)
    canv = np.zeros((P, WID), np.float32)
    if n <= 1:
        return canv
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                  stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
    m = (lab[y:y + h, x:x + w] == i).astype(np.uint8) * 255
    s = max(h, w)
    sq = np.zeros((s, s), np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = m
    q = cv2.resize(sq, (28, 28), interpolation=cv2.INTER_LINEAR)
    canv[14:42, (WID-28)//2:(WID-28)//2+28] = q.astype(np.float32) / 255.0
    return canv


def patch(dom: np.ndarray, xc: float, base: float) -> np.ndarray:
    """2-channel 56x56 patch around (xc, baseline)."""
    H, W = dom.shape
    x0, y0 = int(round(xc)) - SIDE, int(round(base)) - UP
    ch0 = np.zeros((P, WID), np.float32)
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(W, x0 + WID), min(H, y0 + P)
    ch0[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = dom[ys0:ys1, xs0:xs1]
    wx0, wy0 = int(round(xc)) - 16, int(round(base)) - UP
    win = np.zeros((54, 32), np.float32)
    wxs0, wys0 = max(0, wx0), max(0, wy0)
    wxs1, wys1 = min(W, wx0 + 32), min(H, wy0 + 54)
    win[wys0 - wy0:wys1 - wy0, wxs0 - wx0:wxs1 - wx0] = dom[wys0:wys1, wxs0:wxs1]
    m = (win > 0.78).astype(np.uint8)  # lines cap at dom 0.75: never pass
    ch1 = _shape_channel(m)
    return np.stack([ch0, ch1])


# case-folded identity: 0-9 digits, then a-z  (36 classes)
ID36 = string.digits + string.ascii_lowercase
CASE_LOWER, CASE_UPPER, CASE_DIGIT = 0, 1, 2


def id36_of(ch: str) -> int:
    return ID36.index(ch.lower())


def case_of(ch: str) -> int:
    if ch.isdigit():
        return CASE_DIGIT
    return CASE_UPPER if ch.isupper() else CASE_LOWER


def char_of(id36: int, case: int) -> str:
    c = ID36[id36]
    if c.isdigit():
        return c
    return c.upper() if case == CASE_UPPER else c


class GlyphNet(nn.Module):
    """v5: MULTI-TASK.  One trunk, two heads:
         head_id   -> 36-way case-FOLDED identity (digits + a-z)
         head_case -> 3-way {lower, upper, digit}
    Joint 62-way classification forced one layer to do both jobs and capped
    case accuracy ~70%; splitting them lets identity learn shape only while
    case reads vertical geometry.  v4 weights (single 62-way head) still load
    into a v5 model with case_head ignored?  No — shapes differ, so v5 lives
    in its own file and the solver picks whichever verifies better."""

    def __init__(self, n: int = len(ALPHABET), multitask: bool = True,
                 wide: bool = False):
        super().__init__()
        self.multitask = multitask
        self.wide = wide
        fc = 384 if wide else 192
        self.trunk = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 10, fc), nn.ReLU(True), nn.Dropout(0.25),
        )
        if multitask:
            self.head_id = nn.Linear(fc, len(ID36))
            self.head_case = nn.Linear(fc, 3)
        else:
            self.head = nn.Linear(fc, n)

    def forward(self, x):
        f = self.trunk(x)
        if self.multitask:
            return self.head_id(f), self.head_case(f)
        return self.head(f)


_model = None
_models = {}
_HERE = os.path.dirname(os.path.abspath(__file__))


def get_model(path: str = "glyphnet.pt", quiet: bool = False):
    """Load the trained weights.  The default path resolves NEXT TO THIS
    MODULE, not next to the current working directory: launching the runner
    from a different folder (or via a shortcut) must not silently produce a
    model-less process that answers every captcha with an empty string."""
    global _model
    if path == "glyphnet.pt" and _model is not None:
        return _model
    if path in _models:
        return _models[path]
    if not os.path.isabs(path) and not os.path.exists(path):
        candidate = os.path.join(_HERE, path)
        if os.path.exists(candidate):
            path = candidate
    if not os.path.exists(path):
        if not quiet:
            print(f"[glyphnet] no weights at {path!r}", flush=True)
        return None
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if any(k.startswith("net.") for k in sd):      # legacy v4 weights
            sd = {("head." + k.split(".", 2)[2] if k.startswith("net.16.")
                   else "trunk." + k.split(".", 1)[1]): v
                  for k, v in sd.items()}
        wide = any(k == "trunk.13.weight" and v.shape[0] == 384
                   for k, v in sd.items())
        m = GlyphNet(multitask=any(k.startswith("head_id") for k in sd),
                     wide=wide)
        m.load_state_dict(sd)
        m.eval()
        if path == "glyphnet.pt":
            _model = m
        else:
            _models[path] = m
        return m
    except Exception as e:
        print(f"[glyphnet] WARNING: could not load weights from {path!r}: {e}. "
              f"Local reads will be EMPTY; put your Gemini keys in "
              f"gemini_keys.txt so the second opinion carries the rounds.",
              flush=True)
        return None


def classify(dom: np.ndarray, xc: float, base: float):
    """-> (char, confidence)"""
    m = get_model()
    if m is None:
        return "", 0.0
    x = torch.from_numpy(patch(dom, xc, base)[None]).float()
    with torch.no_grad():
        out = m(x)
        if getattr(m, "multitask", False):
            pi, pc = out[0].softmax(-1)[0], out[1].softmax(-1)[0]
            i, c = int(pi.argmax()), int(pc.argmax())
            return char_of(i, c), float(pi[i]) * float(pc[c])
        p = out.softmax(-1)[0]
    i = int(p.argmax())
    return ALPHABET[i], float(p[i])


def classify_many(dom: np.ndarray, xcs, base: float):
    """ONE batched forward for all glyphs -> [(char, confidence), ...]"""
    m = get_model()
    if m is None or len(xcs) == 0:
        return [("", 0.0)] * len(xcs)
    x = torch.from_numpy(np.stack([patch(dom, xc, base) for xc in xcs])).float()
    with torch.no_grad():
        out = m(x)
        if getattr(m, "multitask", False):
            pi, pc = out[0].softmax(-1), out[1].softmax(-1)
            ii, cc = pi.argmax(1), pc.argmax(1)
            conf = pi.gather(1, ii[:, None])[:, 0] * \
                   pc.gather(1, cc[:, None])[:, 0]
            return [(char_of(int(a), int(b)), float(c))
                    for a, b, c in zip(ii, cc, conf)]
        p = out.softmax(-1)
    idx = p.argmax(1)
    conf = p.gather(1, idx[:, None])[:, 0]
    return [(ALPHABET[int(i)], float(c)) for i, c in zip(idx, conf)]


def classify_many_tta(dom: np.ndarray, xcs, base: float, model=None,
                      jitters=((0.0, 0.0), (-1.5, 0.0), (1.5, 0.0),
                               (0.0, -1.0), (0.0, 1.0))):
    """TTA: average the softmax evidence over sub-pixel x-jitters.
    Rotated/AA glyphs sit near patch-grid boundaries; one forward per jitter
    costs ~1 ms each and removes a whole class of flip-flop errors."""
    m = model if model is not None else get_model()
    if m is None or len(xcs) == 0:
        return [("", 0.0)] * len(xcs)
    outs = []
    with torch.no_grad():
        for jx, jy in jitters:
            x = torch.from_numpy(
                np.stack([patch(dom, xc + jx, base + jy) for xc in xcs])
            ).float()
            o = m(x)
            if getattr(m, "multitask", False):
                pi, pc = o[0].softmax(-1), o[1].softmax(-1)
                outs.append((pi, pc))
            else:
                outs.append((o.softmax(-1), None))
    res = []
    if getattr(m, "multitask", False):
        pi = sum(o[0] for o in outs) / len(outs)
        pc = sum(o[1] for o in outs) / len(outs)
        ii, cc = pi.argmax(1), pc.argmax(1)
        conf = pi.gather(1, ii[:, None])[:, 0] * pc.gather(1, cc[:, None])[:, 0]
        res = [(char_of(int(a), int(b)), float(c))
               for a, b, c in zip(ii, cc, conf)]
    else:
        p = sum(o[0] for o in outs) / len(outs)
        idx = p.argmax(1)
        conf = p.gather(1, idx[:, None])[:, 0]
        res = [(ALPHABET[int(i)], float(c)) for i, c in zip(idx, conf)]
    return res


def train(epochs: int = 8, samples: int = 15000, batch: int = 64) -> None:
    import os
    model = GlyphNet()
    # SAFETY FLOOR: a bad epoch-1 initialization must never overwrite good
    # weights.  Read the current model's accuracy first and only save when the
    # new one actually beats it (this burned three retrains already).
    floor = 0.0
    if os.path.exists("glyphnet.pt"):
        try:
            prev = GlyphNet()
            prev.load_state_dict(torch.load("glyphnet.pt", map_location="cpu",
                                            weights_only=True))
            prev.eval()
            shutil.copy("glyphnet.pt", "glyphnet_prev.pt")
            floor = -1.0            # marker: floor measured after dataset load
            del prev
        except Exception:
            floor = 0.0
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    crit = nn.CrossEntropyLoss()
    cache = "glyph_train_wide.npy"
    if not os.path.exists(cache):
        Xu = np.zeros((samples, 2, P, WID), np.uint8)
        Yu = np.zeros(samples, np.int64)
        i = 0
        print("generating dataset…", flush=True)
        rng = np.random.default_rng(0)
        while i < samples:
            img, truth, boxes = generate(boxes=True)
            dom = contrast_field(img)
            for ch, (xc, base, _sz) in zip(truth, boxes):
                # jitter = inference reality: seg centroids/baselines are off
                # by ~N(0,1)px in x and ~N(-1.4,4.2)px in baseline
                xc_j = xc + rng.uniform(-4.0, 4.0)
                base_j = base + rng.uniform(-6.0, 6.0)
                Xu[i] = np.clip(patch(dom, xc_j, base_j) * 255, 0, 255).astype(np.uint8)
                Yu[i] = ALPHABET.index(ch)
                i += 1
        np.save(cache, Xu)
        np.save("glyph_train_wide_y.npy", Yu)
        del Xu
    M = np.load(cache, mmap_mode="r")
    Y = torch.from_numpy(np.load("glyph_train_wide_y.npy"))
    n = M.shape[0]
    print(f"dataset: {n} patches (memmap)", flush=True)

    if floor < 0:                     # measure the weights we are protecting
        try:
            prev = GlyphNet()
            prev.load_state_dict(torch.load("glyphnet.pt", map_location="cpu",
                                            weights_only=True))
            prev.eval()
            c = 0
            with torch.no_grad():
                for i in range(0, min(n, 3000), 500):
                    idx = torch.arange(i, min(i + 500, n))
                    c += int((prev(get(idx)).argmax(-1) == Y[idx]).sum())
            floor = c / min(n, 3000)
            print(f"safety floor: existing weights score {floor:.2%} on this "
                  f"dataset — new weights must beat it", flush=True)
            del prev
        except Exception:
            floor = 0.0

    def get(idx):
        return torch.from_numpy(np.array(M[idx], dtype=np.float32) / 255.0)

    best = 0.0
    for ep in range(epochs):
        perm = torch.randperm(n)
        model.train()
        tot = 0.0
        t0 = time.time()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = crit(model(get(idx)), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, n, 500):
                correct += int((model(get(torch.arange(i, min(i + 500, n)))).argmax(-1)
                                == Y[i:i + 500]).sum())
        acc = correct / n
        print(f"epoch {ep+1}/{epochs} loss {tot/n:.4f} train-acc {acc:.2%} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if acc > best:
            best = acc
            if acc > floor:
                torch.save(model.state_dict(), "glyphnet.pt")
                print(f"  saved ({acc:.2%} > floor {floor:.2%})", flush=True)
            else:
                print(f"  NOT saved: {acc:.2%} <= floor {floor:.2%} "
                      f"(existing weights kept)", flush=True)
    Xv, Yv = [], []
    rng = np.random.default_rng(123)
    for _ in range(400):
        img, truth, boxes = generate(boxes=True)
        dom = contrast_field(img)
        for ch, (xc, base, _sz) in zip(truth, boxes):
            Xv.append(patch(dom, xc + rng.uniform(-4.0, 4.0),
                            base + rng.uniform(-6.0, 6.0)))   # inference-like
            Yv.append(ALPHABET.index(ch))
    Xv = torch.from_numpy(np.stack(Xv)).float(); Yv = torch.tensor(Yv)
    model.eval()
    cv = 0
    with torch.no_grad():
        for i in range(0, Xv.shape[0], 500):
            cv += int((model(Xv[i:i+500]).argmax(-1) == Yv[i:i+500]).sum())
    print(f"held-out glyph accuracy: {cv / Xv.shape[0]:.2%}", flush=True)


def train_v5(epochs: int = 12, samples: int = 24000, batch: int = 64,
             path: str = "glyphnet5.pt") -> None:
    """Multi-task training on the THREE bars at once:
    easy (clean binary) / hard (designed bg, 4-8 chars) / real (page replica).
    Identity head learns case-folded shape; case head learns vertical
    geometry.  Saves only when better than the weights already on disk."""
    import os
    cache, cy1, cy2 = "glyph_train_v5.npy", "glyph_train_v5_yid.npy", \
                      "glyph_train_v5_ycase.npy"
    if not os.path.exists(cache):
        X = np.zeros((samples, 2, P, WID), np.uint8)
        Yid = np.zeros(samples, np.int64)
        Ycase = np.zeros(samples, np.int64)
        i = 0
        print("generating mixed-style dataset...", flush=True)
        rng = np.random.default_rng(0)
        while i < samples:
            style = ("easy", "hard", "real")[i % 3]
            img, truth, boxes = generate(style=style, boxes=True)
            dom = contrast_field(img)
            for ch, (xc, base, _sz) in zip(truth, boxes):
                if i >= samples:
                    break
                xc_j = xc + rng.uniform(-4.0, 4.0)
                base_j = base + rng.uniform(-6.0, 6.0)
                X[i] = np.clip(patch(dom, xc_j, base_j) * 255, 0, 255)\
                    .astype(np.uint8)
                Yid[i] = id36_of(ch)
                Ycase[i] = case_of(ch)
                i += 1
        np.save(cache, X)
        np.save(cy1, Yid)
        np.save(cy2, Ycase)
        del X
    M = np.load(cache, mmap_mode="r")
    Yid = torch.from_numpy(np.load(cy1))
    Ycase = torch.from_numpy(np.load(cy2))
    n = M.shape[0]
    print(f"v5 dataset: {n} patches (memmap)", flush=True)

    def get(idx):
        return torch.from_numpy(np.array(M[idx], dtype=np.float32) / 255.0)

    floor = 0.0
    if os.path.exists(path):
        try:
            sd = torch.load(path, map_location="cpu", weights_only=True)
            prev = GlyphNet(multitask=any(k.startswith("head_id") for k in sd))
            prev.load_state_dict(sd)
            prev.eval()
            c = 0
            m0 = min(n, 3000)
            with torch.no_grad():
                for i in range(0, m0, 500):
                    idx = torch.arange(i, min(i + 500, n))
                    li, lc = prev(get(idx))
                    pred = [char_of(a, b) for a, b in
                            zip(li.argmax(-1).tolist(),
                                lc.argmax(-1).tolist())]
                    true = [char_of(a, b) for a, b in
                            zip(Yid[idx].tolist(), Ycase[idx].tolist())]
                    c += sum(p == t for p, t in zip(pred, true))
            floor = c / m0
            print(f"safety floor: existing v5 weights {floor:.2%} — new "
                  f"epochs must beat it to save", flush=True)
            del prev
        except Exception:
            floor = 0.0

    model = GlyphNet(multitask=True)
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    crit_id = nn.CrossEntropyLoss()
    crit_case = nn.CrossEntropyLoss()
    best = 0.0
    restarts = 0
    ep = -1
    while ep < epochs - 1:
        ep += 1
        perm = torch.randperm(n)
        model.train()
        tot = 0.0
        t0 = time.time()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            li, lc = model(get(idx))
            loss = crit_id(li, Yid[idx]) + 0.5 * crit_case(lc, Ycase[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, n, 500):
                idx = torch.arange(i, min(i + 500, n))
                li, lc = model(get(idx))
                pred = [char_of(a, b) for a, b in
                        zip(li.argmax(-1).tolist(), lc.argmax(-1).tolist())]
                true = [char_of(a, b) for a, b in zip(Yid[idx].tolist(),
                                                      Ycase[idx].tolist())]
                correct += sum(p == t for p, t in zip(pred, true))
        acc = correct / n
        print(f"epoch {ep+1}/{epochs} loss {tot/n:.4f} char-acc {acc:.2%} "
              f"({time.time()-t0:.0f}s)", flush=True)
        # DEAD-INIT TRIPWIRE: a fresh trunk sometimes lands in a flat-loss
        # basin (~uniform predictions).  One bad epoch 1 is not proof, but a
        # sub-10% epoch 1 is: re-roll the weights and redo it (max 4 rolls).
        if ep == 0 and acc < 0.10 and restarts < 3:
            restarts += 1
            print(f"  dead init ({acc:.2%}) — re-rolling weights, "
                  f"attempt {restarts+1}", flush=True)
            model = GlyphNet(multitask=True)
            opt = torch.optim.Adam(model.parameters(), 2e-3)
            ep = -1
            continue
        if acc > best:
            best = acc
            if acc > floor:
                torch.save(model.state_dict(), path)
                print(f"  saved ({acc:.2%} > floor {floor:.2%})", flush=True)
            else:
                print(f"  NOT saved: {acc:.2%} <= floor {floor:.2%}",
                      flush=True)

    # held-out per style, inference-like jitter
    model.eval()
    for style in ("easy", "hard", "real"):
        Xv, Yv = [], []
        rng = np.random.default_rng(123)
        for _ in range(120):
            img, truth, boxes = generate(style=style, boxes=True)
            dom = contrast_field(img)
            for ch, (xc, base, _sz) in zip(truth, boxes):
                Xv.append(patch(dom, xc + rng.uniform(-4, 4),
                                base + rng.uniform(-6, 6)))
                Yv.append(ch)
        Xv = torch.from_numpy(np.stack(Xv)).float()
        cv = 0
        with torch.no_grad():
            for i in range(0, Xv.shape[0], 500):
                li, lc = model(Xv[i:i + 500])
                pred = [char_of(a, b) for a, b in
                        zip(li.argmax(-1).tolist(), lc.argmax(-1).tolist())]
                cv += sum(p == t for p, t in zip(pred, Yv[i:i + 500]))
        print(f"held-out {style}: {cv / Xv.shape[0]:.2%}", flush=True)


def train_v6(epochs: int = 30, samples: int = 40000, batch: int = 128,
             path: str = "glyphnet6.pt", init: str = "glyphnet5.pt") -> None:
    """SCALE-UP fine-tune: start from the v5 weights, 60k patches, balanced
    classes, online augmentation, case-weighted loss, cosine LR.  The bars
    (92/85 exact) need ~99% per-char; 24k patches plateaued at ~70%."""
    import os
    cache, cy1, cy2 = "glyph_train_v6.npy", "glyph_train_v6_yid.npy", \
                      "glyph_train_v6_ycase.npy"
    if not os.path.exists(cache):
        X = np.zeros((samples, 2, P, WID), np.uint8)
        Yid = np.zeros(samples, np.int64)
        Ycase = np.zeros(samples, np.int64)
        i = 0
        print("generating v6 dataset...", flush=True)
        rng = np.random.default_rng(7)
        while i < samples:
            style = ("easy", "hard", "real")[i % 3]
            img, truth, boxes = generate(style=style, boxes=True)
            dom = contrast_field(img)
            for ch, (xc, base, _sz) in zip(truth, boxes):
                if i >= samples:
                    break
                X[i] = np.clip(patch(dom, xc + rng.uniform(-4, 4),
                                     base + rng.uniform(-6, 6)) * 255,
                               0, 255).astype(np.uint8)
                Yid[i] = id36_of(ch)
                Ycase[i] = case_of(ch)
                i += 1
        np.save(cache, X)
        np.save(cy1, Yid)
        np.save(cy2, Ycase)
        del X
    # resident uint8 tensor (zero-copy from the npy): fp16/float32 residents
    # or memmap fancy-indexing both blew the 2 GB sandbox RSS (OOM-kills)
    Xb = torch.from_numpy(np.load(cache))          # shares the numpy buffer
    Yid = torch.from_numpy(np.load(cy1))
    Ycase = torch.from_numpy(np.load(cy2))
    n = Xb.shape[0]
    print(f"v6 dataset: {n} patches (resident uint8)", flush=True)

    def get(idx):
        return Xb[idx].float().div_(255.0)

    # class-balanced sampler: rare chars (q j z digits) otherwise starve
    cnt = np.bincount(Yid.numpy(), minlength=len(ID36)).astype(float) + 1.0
    w = torch.from_numpy(1.0 / cnt[Yid.numpy()])
    sampler = torch.utils.data.BatchSampler(
        torch.utils.data.WeightedRandomSampler(
            w, num_samples=n, replacement=True), batch, drop_last=False)

    model = GlyphNet(multitask=True)
    if os.path.exists(init):          # fine-tune, never scratch
        sd = torch.load(init, map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        print(f"init from {init}", flush=True)
    opt = torch.optim.Adam(model.parameters(), 8e-4)
    crit = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=1e-5)
    floor = 0.0
    if os.path.exists(path):
        try:
            sd = torch.load(path, map_location="cpu", weights_only=True)
            prev = GlyphNet(multitask=True)
            prev.load_state_dict(sd)
            prev.eval()
            c = 0
            m0 = min(n, 3000)
            with torch.no_grad():
                for i in range(0, m0, 500):
                    idx = torch.arange(i, min(i + 500, n))
                    li, lc = prev(get(idx))
                    pred = [char_of(a, b) for a, b in
                            zip(li.argmax(-1).tolist(),
                                lc.argmax(-1).tolist())]
                    true = [char_of(a, b) for a, b in
                            zip(Yid[idx].tolist(), Ycase[idx].tolist())]
                    c += sum(p == t for p, t in zip(pred, true))
            floor = c / m0
            print(f"safety floor: {floor:.2%}", flush=True)
            del prev
        except Exception:
            floor = 0.0

    best = 0.0
    for ep in range(epochs):
        model.train()
        tot = 0.0
        t0 = time.time()
        for idx in sampler:
            idx = torch.as_tensor(idx, dtype=torch.long)
            x = get(idx)
            # online aug: sub-pixel rolls + gamma on the contrast channel
            if np.random.rand() < 0.7:
                dx, dy = np.random.randint(-1, 2, 2)
                x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(2, 3))
            if np.random.rand() < 0.4:
                gm = np.random.choice((0.8, 1.25))
                x = x.clone()
                x[:, 0] = x[:, 0].clamp(0, 1) ** gm
            x = x + 0.02 * torch.rand_like(x) - 0.01
            li, lc = model(x)
            loss = crit(li, Yid[idx]) + 1.5 * crit(lc, Ycase[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, n, 1000):
                idx = torch.arange(i, min(i + 1000, n))
                li, lc = model(get(idx))
                pred = [char_of(a, b) for a, b in
                        zip(li.argmax(-1).tolist(), lc.argmax(-1).tolist())]
                true = [char_of(a, b) for a, b in zip(Yid[idx].tolist(),
                                                      Ycase[idx].tolist())]
                correct += sum(p == t for p, t in zip(pred, true))
        acc = correct / n
        print(f"epoch {ep+1}/{epochs} loss {tot/n:.4f} char-acc {acc:.2%} "
              f"lr {sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)",
              flush=True)
        if acc > best:
            best = acc
            if acc > floor:
                torch.save(model.state_dict(), path)
                print(f"  saved ({acc:.2%} > floor {floor:.2%})", flush=True)

    model.eval()
    for style in ("easy", "hard", "real"):
        Xv, Yv = [], []
        rng = np.random.default_rng(123)
        for _ in range(120):
            img, truth, boxes = generate(style=style, boxes=True)
            dom = contrast_field(img)
            for ch, (xc, base, _sz) in zip(truth, boxes):
                Xv.append(patch(dom, xc + rng.uniform(-4, 4),
                                base + rng.uniform(-6, 6)))
                Yv.append(ch)
        Xv = torch.from_numpy(np.stack(Xv)).float()
        cv = 0
        with torch.no_grad():
            for i in range(0, Xv.shape[0], 500):
                li, lc = model(Xv[i:i + 500])
                pred = [char_of(a, b) for a, b in
                        zip(li.argmax(-1).tolist(), lc.argmax(-1).tolist())]
                cv += sum(p == t for p, t in zip(pred, Yv[i:i + 500]))
        print(f"held-out {style}: {cv / Xv.shape[0]:.2%}", flush=True)


def train_v7(epochs: int = 30, patches_per_epoch: int = 24000, batch: int = 128,
             path: str = "glyphnet7.pt", init: str = "glyphnet6.pt") -> None:
    """ON-THE-FLY training: every epoch regenerates its patches with FRESH
    anchor jitter, so the dataset is effectively infinite and never lives in
    RAM as one block (three OOM-kills taught us: no resident 40k tensor, no
    memmap fancy-indexing).  RAM stays at the ~600 MB torch+cv2 baseline."""
    import os
    model = GlyphNet(multitask=True)
    init = init if os.path.exists(init) else "glyphnet5.pt"
    if os.path.exists(init):
        model.load_state_dict(torch.load(init, map_location="cpu",
                                         weights_only=True))
        print(f"init from {init}", flush=True)
    opt = torch.optim.Adam(model.parameters(), 8e-4)
    crit = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=1e-5)
    floor = 0.0
    if os.path.exists(path):
        try:
            sd = torch.load(path, map_location="cpu", weights_only=True)
            prev = GlyphNet(multitask=True)
            prev.load_state_dict(sd)
            prev.eval()
            floor = _quick_acc(prev, 2000)
            print(f"safety floor: {floor:.2%}", flush=True)
            del prev
        except Exception:
            floor = 0.0

    steps = max(1, patches_per_epoch // batch)
    best = 0.0
    seed = 4242
    for ep in range(epochs):
        model.train()
        tot = 0.0
        seen = 0
        t0 = time.time()
        for _s in range(steps):
            seed += 1
            x, yid, ycase = _fresh_batch(batch, seed)
            if np.random.rand() < 0.7:
                dx, dy = int(np.random.randint(-1, 2)), int(np.random.randint(-1, 2))
                x = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
            if np.random.rand() < 0.4:
                gm = np.random.choice((0.8, 1.25))
                x = x.clone()
                x[:, 0] = x[:, 0].clamp(0, 1) ** gm
            x = x + 0.02 * torch.rand_like(x) - 0.01
            li, lc = model(x)
            loss = crit(li, yid) + 1.5 * crit(lc, ycase)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(yid)
            seen += len(yid)
        sched.step()
        acc = _quick_acc(model, 4000, seed=999)
        print(f"epoch {ep+1}/{epochs} loss {tot/seen:.4f} char-acc {acc:.2%} "
              f"lr {sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)",
              flush=True)
        if ep == 0 and acc < 0.10:
            print("  dead init — re-rolling", flush=True)
            model = GlyphNet(multitask=True)
            opt = torch.optim.Adam(model.parameters(), 8e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=epochs, eta_min=1e-5)
            continue
        if acc > best:
            best = acc
            if acc > floor:
                torch.save(model.state_dict(), path)
                print(f"  saved ({acc:.2%} > floor {floor:.2%})", flush=True)

    model.eval()
    for style in ("easy", "hard", "real"):
        Xv, Yv = [], []
        rng = np.random.default_rng(123)
        for _ in range(120):
            img, truth, boxes = generate(style=style, boxes=True)
            dom = contrast_field(img)
            for ch, (xc, base, _sz) in zip(truth, boxes):
                Xv.append(patch(dom, xc + rng.uniform(-4, 4),
                                base + rng.uniform(-6, 6)))
                Yv.append(ch)
        Xv = torch.from_numpy(np.stack(Xv)).float()
        cv = 0
        with torch.no_grad():
            for i in range(0, Xv.shape[0], 500):
                li, lc = model(Xv[i:i + 500])
                pred = [char_of(a, b) for a, b in
                        zip(li.argmax(-1).tolist(), lc.argmax(-1).tolist())]
                cv += sum(p == t for p, t in zip(pred, Yv[i:i + 500]))
        print(f"held-out {style}: {cv / Xv.shape[0]:.2%}", flush=True)


def train_v8(epochs: int = 40, patches_per_epoch: int = 24000, batch: int = 128,
             path: str = "glyphnet8.pt", init: str = "glyphnet7.pt") -> None:
    """WIDER trunk (fc 384) fine-tuned from v7 (conv layers carry over, the
    fc/head re-init).  v7 plateaued ~96% train / 91% suite per-char; capacity
    is the next lever after data freshness."""
    import os
    model = GlyphNet(multitask=True, wide=True)
    if os.path.exists(init):
        sd = torch.load(init, map_location="cpu", weights_only=True)
        own = model.state_dict()
        sd = {k: v for k, v in sd.items()
              if k in own and own[k].shape == v.shape}
        miss, _unexp = model.load_state_dict(sd, strict=False)
        print(f"init from {init}: {len(sd)} keys carried, "
              f"{len(miss)} re-init", flush=True)
    opt = torch.optim.Adam(model.parameters(), 6e-4)
    crit = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=1e-5)
    steps = max(1, patches_per_epoch // batch)
    best = 0.0
    seed = 99117
    for ep in range(epochs):
        model.train()
        tot = 0.0
        seen = 0
        t0 = time.time()
        for _s in range(steps):
            seed += 1
            x, yid, ycase = _fresh_batch(batch, seed)
            if np.random.rand() < 0.7:
                dx, dy = int(np.random.randint(-1, 2)), int(np.random.randint(-1, 2))
                x = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
            if np.random.rand() < 0.4:
                gm = np.random.choice((0.8, 1.25))
                x = x.clone()
                x[:, 0] = x[:, 0].clamp(0, 1) ** gm
            x = x + 0.02 * torch.rand_like(x) - 0.01
            li, lc = model(x)
            loss = crit(li, yid) + 1.5 * crit(lc, ycase)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(yid)
            seen += len(yid)
        sched.step()
        acc = _quick_acc(model, 4000, seed=999)
        print(f"epoch {ep+1}/{epochs} loss {tot/seen:.4f} char-acc {acc:.2%} "
              f"lr {sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)",
              flush=True)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), path)
            print(f"  saved ({acc:.2%})", flush=True)
    model.eval()
    for style in ("easy", "hard", "real"):
        Xv, Yv = [], []
        rng = np.random.default_rng(123)
        for _ in range(120):
            img, truth, boxes = generate(style=style, boxes=True)
            dom = contrast_field(img)
            for ch, (xc, base, _sz) in zip(truth, boxes):
                Xv.append(patch(dom, xc + rng.uniform(-4, 4),
                                base + rng.uniform(-6, 6)))
                Yv.append(ch)
        Xv = torch.from_numpy(np.stack(Xv)).float()
        cv = 0
        with torch.no_grad():
            for i in range(0, Xv.shape[0], 500):
                li, lc = model(Xv[i:i + 500])
                pred = [char_of(a, b) for a, b in
                        zip(li.argmax(-1).tolist(), lc.argmax(-1).tolist())]
                cv += sum(p == t for p, t in zip(pred, Yv[i:i + 500]))
        print(f"held-out {style}: {cv / Xv.shape[0]:.2%}", flush=True)


def _fresh_batch(batch: int, seed: int):
    """-> (x float32 [B,2,P,WID], yid, ycase) generated live, mixed styles."""
    random.seed(seed)
    xs, yid, ycase = [], [], []
    rng = np.random.default_rng(seed)
    while len(xs) < batch:
        style = ("easy", "hard", "real")[len(xs) % 3]
        img, truth, boxes = generate(style=style, boxes=True)
        dom = contrast_field(img)
        for ch, (xc, base, _sz) in zip(truth, boxes):
            if len(xs) >= batch:
                break
            xs.append(patch(dom, xc + rng.uniform(-4, 4),
                            base + rng.uniform(-6, 6)))
            yid.append(id36_of(ch))
            ycase.append(case_of(ch))
    return (torch.from_numpy(np.stack(xs)).float(),
            torch.tensor(yid, dtype=torch.long),
            torch.tensor(ycase, dtype=torch.long))


def _quick_acc(model, m, seed=7):
    model.eval()
    c = 0
    done = 0
    with torch.no_grad():
        while done < m:
            x, yid, ycase = _fresh_batch(min(500, m - done), seed)
            seed += 1
            li, lc = model(x)
            pred = [char_of(a, b) for a, b in
                    zip(li.argmax(-1).tolist(), lc.argmax(-1).tolist())]
            true = [char_of(a, b) for a, b in
                    zip(yid.tolist(), ycase.tolist())]
            c += sum(p == t for p, t in zip(pred, true))
            done += len(x)
    return c / m


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train5":
        train_v5(int(sys.argv[2]) if len(sys.argv) > 2 else 12,
                 int(sys.argv[3]) if len(sys.argv) > 3 else 24000)
    if len(sys.argv) > 1 and sys.argv[1] == "train8":
        train_v8(int(sys.argv[2]) if len(sys.argv) > 2 else 40,
                 int(sys.argv[3]) if len(sys.argv) > 3 else 24000)
    if len(sys.argv) > 1 and sys.argv[1] == "train7":
        train_v7(int(sys.argv[2]) if len(sys.argv) > 2 else 30,
                 int(sys.argv[3]) if len(sys.argv) > 3 else 24000)
    if len(sys.argv) > 1 and sys.argv[1] == "train6":
        train_v6(int(sys.argv[2]) if len(sys.argv) > 2 else 30,
                 int(sys.argv[3]) if len(sys.argv) > 3 else 60000)
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 8,
              int(sys.argv[3]) if len(sys.argv) > 3 else 15000)
    else:
        print("model loaded:", get_model() is not None)
