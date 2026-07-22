"""
Pull a large, accurate Statcast dataset for the MLB predictor.

Strategy for maximum accuracy: combine a full PRIOR season (2025, a large stable
baseline of true talent) with the CURRENT season to date (2026 form). This mirrors
how real projection systems (Marcel/ZiPS/Steamer) blend multiple years.

Downloads month-by-month and appends to the CSV so progress is preserved if the
connection drops. Keeps regular-season games only (game_type == 'R').
"""
import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
from datetime import date
from pybaseball import statcast

OUT_FILE = "mlb_batting_data.csv"

# (label, start, end) monthly windows. Regular season roughly late-Mar -> Sep.
# 2026 is capped at "today" since the season is still in progress.
TODAY = date.today().isoformat()
WINDOWS = [
    # ---- 2025 full regular season ----
    ("2025-03", "2025-03-27", "2025-03-31"),
    ("2025-04", "2025-04-01", "2025-04-30"),
    ("2025-05", "2025-05-01", "2025-05-31"),
    ("2025-06", "2025-06-01", "2025-06-30"),
    ("2025-07", "2025-07-01", "2025-07-31"),
    ("2025-08", "2025-08-01", "2025-08-31"),
    ("2025-09", "2025-09-01", "2025-09-30"),
    # ---- 2026 season to date ----
    ("2026-03", "2026-03-26", "2026-03-31"),
    ("2026-04", "2026-04-01", "2026-04-30"),
    ("2026-05", "2026-05-01", "2026-05-31"),
    ("2026-06", "2026-06-01", "2026-06-30"),
    ("2026-07", "2026-07-01", TODAY),
]

# Don't request windows that start in the future.
WINDOWS = [w for w in WINDOWS if w[1] <= TODAY]

print(f"Building dataset -> {OUT_FILE}")
print(f"Today is {TODAY}. Pulling {len(WINDOWS)} monthly windows.\n")

# Start clean so re-runs don't duplicate rows.
if os.path.exists(OUT_FILE):
    os.remove(OUT_FILE)

total_rows = 0
wrote_header = False

for label, start, end in WINDOWS:
    print(f"⏳ {label}  ({start} → {end}) ...", end=" ", flush=True)
    try:
        chunk = statcast(start_dt=start, end_dt=end)
    except Exception as e:
        print(f"skipped (error: {repr(e)[:80]})")
        continue

    if chunk is None or chunk.empty:
        print("no games")
        continue

    # Regular season only — keeps player-talent estimates clean.
    if "game_type" in chunk.columns:
        chunk = chunk[chunk["game_type"] == "R"]

    if chunk.empty:
        print("no regular-season rows")
        continue

    chunk.to_csv(OUT_FILE, mode="a", index=False, header=not wrote_header)
    wrote_header = True
    total_rows += len(chunk)
    print(f"+{len(chunk):,} rows  (running total {total_rows:,})")

print("\n" + "=" * 50)
if total_rows == 0:
    print("❌ No data pulled. Check your connection and try again.")
else:
    pas = pd.read_csv(OUT_FILE, low_memory=False)["events"].notna().sum()
    print(f"✅ Saved {total_rows:,} pitches ({pas:,} plate appearances) to {OUT_FILE}")
print("=" * 50)
