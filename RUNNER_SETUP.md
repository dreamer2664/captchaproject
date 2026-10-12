# LIVE RUNNER — Windows setup (read + type on YOUR pc)

## 1. Install
- Python 3.11+ from python.org (tick "Add to PATH")
- In the project folder:
      pip install -r requirements.txt
  (smaller torch: `pip install torch --index-url https://download.pytorch.org/whl/cpu` first)
- `glyphnet.pt` (trained weights) must sit next to local_runner.py — it's in the zip.

## 2. Calibrate (once) — ONE BIG REGION
      python local_runner.py calibrate
- Frame a GENEROUS area that always contains the captcha image(s) AND the
  answer text box — the software now FINDS everything itself every round:
    * image size changes? handled (auto-detected + rescaled to model size)
    * image moves around? handled
    * TWO captchas (dual mode)? handled — answer is typed "text1 text2"
      with the space, exactly as the page expects
    * page furniture (titles, labels, timer bar, buttons)? ignored — the
      runner learns static content after round 1 and skips it
- Optional fallback text-box point (only used if auto-detect ever fails).
- Submit: Enter (default, fastest) or calibrated button click.
- It asks "Characters PER captcha? [6]" — enforced everywhere; wrong-length
  reads trigger the fallbacks instead of being typed.
- Saves `runner_config.json` (keep it when re-extracting new zips!).

## 3. Test without typing
      python local_runner.py once
- Prints: boxes found, text-box position, per-captcha read + source,
  final ANSWER (space-joined in dual mode) + latency.
- Saves `last_capture.png` (raw grab) and `last_debug.png`
  (green boxes = detected captchas with their reads, blue circle = detected
  text box).
- **When a read is wrong, send me those two PNGs** — that's how I tune the
  model to YOUR exact captcha (noise strength, fonts, scale).

## 4. Gemini keys (optional second opinion)
- Keys one per line in `gemini_keys.txt` (or env `GEMINI_API_KEYS=k1,k2,...`).
- Model: env `GEMINI_MODEL` (default `gemini-2.5-flash`).
- The 15 s law: Gemini is called ONLY when the local confidence is low AND
  remaining time > its timeout + typing + margin. Hard deadline: each
  request's timeout shrinks to fit the budget (even-sliced across attempts,
  so one slow call can't eat the round); if it stalls, the local answer
  goes in anyway. Rounds are never gambled, and blank/expired regions
  never reach the network.
- With keys loaded, the second opinion VOTES: 2-3 reads, length-checked,
  majority wins, early exit on the first agreement. Wrong-length reads are
  discarded. Measured worst case (20 s-latency network, dual mode): the
  whole solve stays inside 13 s — the 15 s round law holds.

## 5. Run the bot
      python local_runner.py run
- Polls the region 5×/s, detects new rounds by image hash,
  reads (~0.1 s typical), clicks the box, Ctrl+A/Del, types with 30-70 ms
  human jitter, submits (Enter or click per calibration).
- If a read somehow takes longer than the round allows, the submit is
  SKIPPED (the round is already dead — no point burning time typing).
- Console logs per round: text, reader, read-time, submit-time, warnings.
- ABORT ANY TIME: mouse into a screen corner (pyautogui failsafe).

## Accuracy bars & how they are measured
- `python3 suite_read.py 100`  — local reader only, three suites:
    easy  = clean binary cards (white/black, black/white)      bar: 92% exact
    hard  = designed backgrounds (objects, patterns), mixed
            fonts/colors/sizes, 4-8 chars                      bar: 85% exact
    real  = replica of your real page's cards
- `python3 suite_detect.py 100` — finder on the full real-page clone:
    captcha boxes (single+dual), red-ring textbox, 14s-notice immunity.
- `python3 suite_e2e.py 60`     — detect + read + space-join on whole pages.
- The reader is a multi-task CNN (36-way case-folded identity + 3-way case
  head) trained on all three styles at once with segmentation jitter.

## Troubleshooting
- Wrong boxes detected on the FIRST round only: normal — the static-
  furniture filter needs one round of history; from round 2 it's clean.
- Reads full of junk chars: captcha noise stronger than the model knows —
  send last_capture.png + last_debug.png for a retrain on your exact style.
- "no read — waiting": region doesn't contain the captcha right now
  (page scrolled/moved) — re-calibrate the big region.
- Slow first round only: torch model load (~2-5 s once per process).
- Dual mode IS implemented: both captchas are found, read, and
  joined with a SPACE between them ("abc123 xyz789").
- After a page restyle: re-calibrate, and send a screenshot for retraining.

## Accuracy log (sandbox suites, local reader only — no Gemini keys here)

| date | change | easy exact | hard exact | real exact | e2e exact |
|---|---|---|---|---|---|
| oct10 am | v4 weights, broken generator | 7 | 0 | 0 | - |
| oct10 pm | RGBA-alpha fix + anchor fix + dom-ink seg | 30 | 13 | 13 | 3 |
| oct10 pm | + dom sign-after-ring fix (REAL pages were polarity-dead!) | - | - | - | rising |
| oct10 eve | v5 multitask (14 ep, 24k) | 12 | 5 | 7 | 2 |
| oct10 night | v7 on-the-fly 30ep + geometry cues | 72 | 27 | 30 | 8 |
| training | v8 wide fc384, 40 ep, from v7 | 73 | 27 | 31 | - |
| oct11 | v9 whole-ch1 + case loss x2, stopped ep7/50 | 75 | 33 | 35 | - |
| oct11 | v10 inference-estimator anchors, 30 ep, from v9 | 89 | 47 | 44 | - |
| oct11 | + POSTPASS off (model beat every hand rule) | 89 | 48 | 44 | - |
| oct11 | + generator ambiguity fix, O0 mini-pass, W-split gate | 91 | 42 | 44 | - |
| training | v11 hard-weighted (50% hard), case x3, morph aug | ? | ? | ? | ? |

v10 held-out per-char (patches anchored where inference anchors them):
easy 98.75 / hard 88.34 / real 88.75.
glyphnet.pt = v10 weights.  Older: glyphnet9.pt, glyphnet8.pt, glyphnet7.pt.

Two changes mattered more than any training run:
1. ANCHOR MATCH (v10): training patches used to be cut at the true glyph
   centre; inference cuts at the nearest component ink centroid + estimated
   baseline.  Closing that train/serve skew was worth ~+15 exact.
2. DELETE THE HAND RULES (POSTPASS=False): the case/geometry/template post
   passes were written for the weak v7 head.  Against v10 they only misfired
   -- A/B on 60 seeds per style: raw beat full by +8 easy, +8 hard, +9 real.
   The code stays behind POSTPASS in solver.py for pre-v10 checkpoints.
Ensembling v10+v9 is now a wash (same exact, 2x latency) so it defaults off;
GLYPHNET_ENS=1 turns it back on.

Suite numbers are means over 3 x 100 seeds (offsets 0 / 100003 / 200007):
easy 90.0, hard 42.0, real 49.3 -- the single 100-seed run above is one draw
of that, so treat +-4 on hard as noise, not progress.

Bugs fixed oct10 (all silently catastrophic before):
1. ImageDraw on RGBA writes alpha raw -> designs/noise rendered SOLID (looked
   like ink). Now drawn on transparent overlays + true alpha_composite.
2. Glyph paste off by half advance vs recorded anchor -> train/infer mismatch.
3. dom sign computed BEFORE frame-ring strip -> any loose crop (page bg ring)
   flipped polarity -> dead dom -> '111111' garbage. THIS killed real runs.
4. Legacy v4 weights (net.* keys) wouldn't load into v5 class -> remap added.
