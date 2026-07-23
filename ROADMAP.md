# 🗺️ Roadmap — future improvements

The model (v1.1) is live, validated, and honest. There's no single "best" recipe —
improvement is a continuous loop. Each change below should be validated the same way
we've done everything: **add it → re-run `backtest.py` → keep it only if accuracy/Brier
improves**, and watch the live track record confirm it over time.

## Current honest performance (v1.1, 600-game backtest)
- Winner accuracy ~56% (beats "always home" ~54%)
- Brier ~0.244 (beats naive baseline 0.247) — **probabilities are calibrated**
- Run totals unbiased (~0)
- Strength is calibration, not a flashy accuracy number.

## Biggest lever: the ML phase (Phase 4)
The original goal. We built a validated *simulation* first (the right call). Next is a
**machine-learning layer**: feed all the variables (rates, platoon, park, weather, bullpen,
recency, home-field) into a model that *learns* their weights and interactions. Then use the
backtest + live track record to prove whether ML beats the simulation. Likely the biggest
remaining accuracy gain.

## Other improvements (any order, all optional)
- **Keep data fresh** — CSV is a snapshot; re-run `fetch_data.py` periodically for current form.
- **Times-through-order penalty** — starters get worse the 3rd time through a lineup (real, unmodeled).
- **Better bullpen** — specific relievers, closer/leverage usage, inherited-runner ERA accounting.
- **Confirmed lineups / injuries** — is the star actually playing tonight?
- **Re-tune weights** using the accumulating live track record (recency half-life, shrinkage, home-field, park factors).
- **Revisit team defense** — was tested & rejected (double-counted pitcher stats); could work with a smarter approach.
- **Umpire tendencies** (column exists, unused), stolen bases / speed, handedness-specific park factors.

## Then: expand beyond MLB
The app shell, betting tools, backtest, and track-record system are sport-agnostic. Each new
sport needs its own simulation engine + data source, plugged into the same framework. Also:
a college-baseball mode (needs college data or manual player-rate input) and a "coach mode".

## The discipline (don't skip this)
- Every model change: bump `MODEL_VERSION` in `app.py`, backtest it, and let the live log compare versions on real results.
- Never trust one backtest sample for accuracy (it's noisy ±3%); use the 300–600 game window and watch the live track record.
