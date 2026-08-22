"""
INCREMENTAL data refresh for the MLB predictor.

The full pull (fetch_data.py) re-downloads 2025 + 2026 and takes ~2h. This script
instead pulls ONLY the days missing from mlb_batting_data.csv and appends them, so
it runs in seconds-to-minutes and can go in CI. It reproduces the slim schema
(PA rows, 15 columns) that fetch_data.py's output was historically hand-reduced to
— so it doubles as the reproducible "slim" step that was previously manual.

  python3 refresh_data.py               # catch up from last date in the CSV -> today
  python3 refresh_data.py --start 2026-07-21 --end 2026-08-14   # explicit range

Safe: writes a .bak first, dedupes on (game_pk, at_bat_number), only appends.
"""
import warnings; warnings.filterwarnings("ignore")
import argparse
import shutil
from datetime import date, timedelta
import pandas as pd
from pybaseball import statcast

OUT = "mlb_batting_data.csv"
SLIM_COLS = ["game_date", "batter", "pitcher", "events", "description", "stand",
             "p_throws", "home_team", "away_team", "inning", "inning_topbot",
             "game_pk", "at_bat_number", "post_away_score", "post_home_score"]
INT_COLS = ["batter", "pitcher", "inning", "game_pk", "at_bat_number",
            "post_away_score", "post_home_score"]


def slim(chunk):
    """Reduce a raw Statcast chunk to the committed 15-col PA-only schema."""
    if "game_type" in chunk.columns:
        chunk = chunk[chunk["game_type"] == "R"]           # regular season only
    chunk = chunk[chunk["events"].notna()]                 # one row per plate appearance
    missing = [c for c in SLIM_COLS if c not in chunk.columns]
    if missing:
        raise SystemExit(f"Statcast chunk missing expected columns: {missing}")
    return chunk[SLIM_COLS].copy()


def main(start, end):
    old = pd.read_csv(OUT, low_memory=False)
    old["game_date"] = old["game_date"].astype(str)
    last = old["game_date"].max()
    if start is None:
        start = (pd.Timestamp(last) + timedelta(days=1)).date().isoformat()
    if end is None:
        end = date.today().isoformat()

    print(f"Existing: {len(old):,} PAs through {last}")
    if start > end:
        print(f"Already current (would start at {start}, past {end}). Nothing to do.")
        return

    print(f"Pulling {start} -> {end} (incremental)...")
    chunk = statcast(start_dt=start, end_dt=end)
    if chunk is None or chunk.empty:
        print("No new Statcast rows returned. Nothing to append.")
        return
    new = slim(chunk)
    new["game_date"] = new["game_date"].astype(str)
    print(f"  pulled {len(new):,} new PAs ({new['game_date'].min()} -> {new['game_date'].max()})")

    combined = pd.concat([old, new], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["game_pk", "at_bat_number"], keep="last")
    # Keep the id/score columns as clean integers (no 519242.0), NaN-safe.
    for c in INT_COLS:
        combined[c] = pd.to_numeric(combined[c], errors="coerce").astype("Int64")

    shutil.copyfile(OUT, OUT + ".bak")                     # safety net
    combined.to_csv(OUT, index=False)
    added = len(combined) - len(old)
    print(f"✅ {OUT}: {len(old):,} -> {len(combined):,} PAs "
          f"(+{added:,} new, {before - len(combined):,} dupes dropped)")
    print(f"   now through {combined['game_date'].max()}  (backup: {OUT}.bak)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (default: day after last row)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    a = ap.parse_args()
    main(a.start, a.end)
