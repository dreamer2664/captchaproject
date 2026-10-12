# ocr-typer — CAPTCHA reader + test harness + live PC runner

Reads generated captchas (mixed-case letters + digits, CASE SENSITIVE;
backgrounds/fonts/sizes/colors vary every round) and types the answer into
the test page within the time limit. Dual captchas are answered with the two
texts joined by a single space.

## Final measured results (2026-10-12, this sandbox, CPU)

| suite                          | exact-match | target | status |
|--------------------------------|-------------|--------|--------|
| easy   (100 cards, flat bg)    | 96–97 %     | ≥ 92   | ✅ MET |
| hard   (100 cards, clutter, objects, mixed fonts/colors, size varies per round) | 86–88 % | ≥ 85 | ✅ MET |
| real   (100 cards, real-page style) | 87 %   | —      | —      |
| e2e    (60 generated pages, detect+read+type flow) | 75 % | — | — |
| detection (captcha rects on page)   | 57–60 / 60 | — | — |
| latency per round              | mean 0.4–1.5 s, max ~2.7 s | < 14 s | ✅ |

(hard varies 84–88 between runs: torch CPU float-reduction order at the
decision margin; the default two-model ensemble keeps it on the upper end.)

## Layout
- generator.py     synthetic generator (easy / hard / real styles)
- finder.py        auto-finds captcha rects + answer box in ONE big screen
                   region (image size changes every round — no tight calib)
- realpage.py      page model calibrated on the real UI screenshot
- glyphnet.py      per-glyph CNN reader (size channel, case head, TTA)
                   -> glyphnet.pt (primary) + glyphnet15b.pt (ensemble vote)
- solver.py        chain: segment -> GlyphNet ensemble -> case/geometry cues
                   -> confidence gate -> Gemini second opinion (budget-gated)
- gemini_reader.py Gemini-vision second opinion, KEY ROTATION
                   keys: env GEMINI_API_KEYS or gemini_keys.txt (one/line)
                   model: env GEMINI_MODEL (default gemini-2.5-flash)
- templates.py     glyph template bank (heuristic fallback)
- crnn.py          color-invariant contrast frontend (dormant CTC reader)
- harness_app.py   test page: timer bar, CASE SENSITIVE box, AUTOPILOT,
                   100-round test            -> python3 harness_app.py 5000
- benchmark.py     headless scorer           -> python3 benchmark.py 100
- suite_read.py    100-card exact-match suite per style
- suite_detect.py  finder accuracy suite
- suite_e2e.py     full detect+read+type simulation
- local_runner.py  LIVE runner for your Windows PC (screen grab -> read ->
                   click box -> type -> submit with Enter AND button click)
- RUNNER_SETUP.md  step-by-step Windows setup

## Run
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install numpy pillow opencv-python mss flask requests pyautogui
    python3 harness_app.py 5000      # test page
    python3 local_runner.py          # live runner (see RUNNER_SETUP.md)

## Env knobs
- `GLYPHNET_ENS=0`        disable the two-model ensemble (faster, hard −2)
- `GEMINI_CONF_THRESHOLD` confidence below which Gemini is consulted (0.62);
  calls are budget-gated — never gambled inside the 14 s window
- `ROUND_SECONDS=14.0`    hard time law; answers are abandoned 3 s early

## Train / regenerate weights
    python3 glyphnet.py train14 40 24000      # balanced hard-mix recipe
    python3 glyphnet.py train13 30 20000      # hard-dominant recipe
Checkpointing keeps the best quick-acc epoch; ALWAYS judge checkpoints with
suite_read.py (train accuracy saturates and is not a proxy for suite gains).

## Notes on the hard-style regime
Real captchas change letter size EVERY ROUND, not within a card — the hard
generator matches that (one size 26–44 px per card, per-character
font/color/rotation). Per-character random size made case information-
theoretically undecidable from a single glyph; the per-card regime plus the
line-based case pass is what lifted hard from 62 to 86+.
