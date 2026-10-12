"""Error miner: run the current weights over fresh held-out glyphs per style,
dump the confusion matrix + worst characters, so the next training round can
over-sample exactly the pairs the model flips (0/O, l/I, b/6, case pairs...).

    python3 mine_errors.py [per_style]
"""
import random
import sys
from collections import Counter, defaultdict

import numpy as np

import generator
import glyphnet
from crnn import contrast_field

PER = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def main():
    m = glyphnet.get_model()
    if m is None:
        print("no weights")
        return
    conf = Counter()
    per_char = Counter()
    per_char_bad = Counter()
    case_err = 0
    id_err = 0
    tot = 0
    for style in ("easy", "hard", "real"):
        jit = random.Random(777 + len(style))
        for i in range(PER):
            random.seed(777 + len(style) * 1000 + i)   # generator = global rng
            img, truth, boxes = generator.generate(style=style, boxes=True)
            dom = contrast_field(img)
            base_j = None
            for ch, (xc, base, _sz) in zip(truth, boxes):
                p = glyphnet.patch(dom, xc + jit.uniform(-1.5, 1.5),
                                   base + jit.uniform(-2, 2),
                                   getattr(m, "ch1_whole", False))
                import torch
                with torch.no_grad():
                    out = m(torch.from_numpy(p[None]).float())
                if getattr(m, "multitask", False):
                    pi, pc = out[0].softmax(-1)[0], out[1].softmax(-1)[0]
                    a, c = int(pi.argmax()), int(pc.argmax())
                    got = glyphnet.char_of(a, c)
                else:
                    got = glyphnet.ALPHABET[int(out.softmax(-1)[0].argmax())]
                tot += 1
                per_char[ch] += 1
                if got != ch:
                    per_char_bad[ch] += 1
                    conf[(ch, got)] += 1
                    if ch.lower() == got.lower():
                        case_err += 1
                    else:
                        id_err += 1
    print(f"total {tot}  case-errors {case_err} ({case_err/max(tot,1):.1%})  "
          f"identity-errors {id_err} ({id_err/max(tot,1):.1%})")
    print("\nworst truths (err/total):")
    for ch, bad in per_char_bad.most_common(14):
        print(f"  {ch!r}: {bad}/{per_char[ch]} = {bad/per_char[ch]:.0%}")
    print("\ntop confusions truth->got:")
    for (t, g), n in conf.most_common(20):
        print(f"  {t!r}->{g!r}: {n}")


if __name__ == "__main__":
    main()
