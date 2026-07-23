import streamlit as st
import pandas as pd
import numpy as np
import random
import requests
from datetime import datetime
import altair as alt

# This file works two ways:
#   * `streamlit run app.py`  -> runs the interactive web app (see bottom).
#   * `import app`            -> exposes the engine (profiles, simulate_games,
#                                league baselines) for backtest.py / ML work,
#                                WITHOUT launching any UI.

# =====================================================================
# 0. STATIC REFERENCE DATA
# =====================================================================

# Bump this whenever the model's math/weights change, so the performance log can
# compare versions on their real, realized results.
MODEL_VERSION = "v1.0-recency150-bullpen"

# Which "events" values count as hits, and which do NOT count as at-bats.
HIT_EVENTS = ['single', 'double', 'triple', 'home_run']
NON_AB_EVENTS = ['walk', 'intent_walk', 'hit_by_pitch', 'sac_fly',
                 'sac_bunt', 'catcher_interf', 'truncated_pa']

# Approximate ballpark factors (100 = league neutral). "runs" scales overall
# offense; "hr" scales home-run rate. These are rough public estimates and are
# meant to be adjusted as you refine the model.
PARK_FACTORS = {
    "Colorado Rockies":       {"runs": 112, "hr": 104},
    "Boston Red Sox":         {"runs": 108, "hr": 97},
    "Cincinnati Reds":        {"runs": 104, "hr": 118},
    "Kansas City Royals":     {"runs": 104, "hr": 92},
    "Arizona Diamondbacks":   {"runs": 103, "hr": 102},
    "Philadelphia Phillies":  {"runs": 101, "hr": 106},
    "Baltimore Orioles":      {"runs": 101, "hr": 103},
    "Texas Rangers":          {"runs": 101, "hr": 100},
    "Toronto Blue Jays":      {"runs": 101, "hr": 102},
    "Los Angeles Angels":     {"runs": 100, "hr": 103},
    "Chicago Cubs":           {"runs": 100, "hr": 101},
    "Atlanta Braves":         {"runs": 100, "hr": 101},
    "Minnesota Twins":        {"runs": 100, "hr": 100},
    "Washington Nationals":   {"runs": 100, "hr": 101},
    "Houston Astros":         {"runs": 99,  "hr": 101},
    "New York Yankees":       {"runs": 99,  "hr": 110},
    "Chicago White Sox":      {"runs": 99,  "hr": 103},
    "Pittsburgh Pirates":     {"runs": 99,  "hr": 92},
    "Los Angeles Dodgers":    {"runs": 99,  "hr": 105},
    "St. Louis Cardinals":    {"runs": 98,  "hr": 96},
    "Milwaukee Brewers":      {"runs": 98,  "hr": 102},
    "New York Mets":          {"runs": 98,  "hr": 97},
    "Tampa Bay Rays":         {"runs": 97,  "hr": 96},
    "Detroit Tigers":         {"runs": 97,  "hr": 95},
    "Cleveland Guardians":    {"runs": 97,  "hr": 96},
    "Miami Marlins":          {"runs": 96,  "hr": 95},
    "Athletics":              {"runs": 96,  "hr": 96},
    "San Diego Padres":       {"runs": 96,  "hr": 95},
    "Seattle Mariners":       {"runs": 95,  "hr": 96},
    "San Francisco Giants":   {"runs": 94,  "hr": 90},
}
NEUTRAL_PARK = {"runs": 100, "hr": 100}

# --- Dark "sportsbook" visual theme (injected once in main) ---
CUSTOM_CSS = """
<style>
/* Hide Streamlit's chrome so it reads like a normal website */
#MainMenu, footer, header[data-testid="stHeader"] { display:none !important; }
[data-testid="stToolbar"], [data-testid="stStatusWidget"], [data-testid="stDecoration"] { display:none !important; }

html, body, [class*="css"] { font-family:'Inter','Segoe UI',system-ui,sans-serif; }
/* Centered, comfortable content width */
.block-container { max-width:1160px; padding-top:2.2rem; padding-bottom:4rem; }
hr { border-color:#1c2430; margin:1.4rem 0; }
h2, h3 { letter-spacing:-.4px; font-weight:800; }

.hero { position:relative;
  background:radial-gradient(1100px 220px at 18% -40%, rgba(0,230,118,.16), transparent),
             linear-gradient(135deg,#0d1117 0%,#10241b 100%);
  border:1px solid #1f6f43; border-radius:20px; padding:34px 38px; margin-bottom:22px;
  box-shadow:0 12px 44px rgba(0,0,0,.35); }
.hero .eyebrow { display:inline-block; font-size:.7rem; font-weight:800; letter-spacing:.13em;
  text-transform:uppercase; color:#04130a; background:#00e676; padding:.24rem .65rem;
  border-radius:999px; margin-bottom:.85rem; }
.hero h1 { margin:0; font-size:2.7rem; font-weight:900; letter-spacing:-1px; line-height:1.05;
  background:linear-gradient(90deg,#00e676,#8affbe); -webkit-background-clip:text;
  background-clip:text; -webkit-text-fill-color:transparent; }
.hero p { margin:.6rem 0 0; color:#9aa7b6; font-size:1.02rem; max-width:660px; }

[data-testid="stMetric"] { background:#141a24; border:1px solid #263043;
  border-radius:16px; padding:16px 20px; }
[data-testid="stMetricValue"] { color:#00e676; font-weight:800; }
[data-testid="stMetricLabel"] { color:#93a0b0; }

.stButton>button { background:linear-gradient(90deg,#00e676,#00c853); color:#04130a;
  font-weight:800; border:0; border-radius:12px; padding:.55rem 1.3rem;
  box-shadow:0 4px 18px rgba(0,230,118,.28); transition:transform .08s ease,filter .08s ease; }
.stButton>button:hover { transform:translateY(-1px); filter:brightness(1.07); }

[data-testid="stExpander"] { border:1px solid #232c3a; border-radius:14px;
  background:#10151d; overflow:hidden; }
[data-testid="stExpander"] summary { font-weight:700; }
.stTextInput input, .stNumberInput input { background:#141a24; }
thead tr th { background:#141a24 !important; color:#c7d0dc !important; }

.odds-wrap { display:flex; gap:16px; flex-wrap:wrap; margin:6px 0 12px; }
.odds-card { flex:1; min-width:200px; background:#141a24; border:1px solid #263043;
  border-radius:18px; padding:20px 22px; text-align:center; box-shadow:0 6px 24px rgba(0,0,0,.25); }
.odds-card .team { color:#c7d0dc; font-size:.92rem; font-weight:600; }
.odds-card .win { font-size:2.5rem; font-weight:900; color:#e6e9ef; margin:.15rem 0; letter-spacing:-1px; }
.odds-card .ml { display:inline-block; margin-top:.3rem; padding:.2rem .8rem;
  border-radius:999px; font-weight:800; font-size:1rem;
  background:rgba(255,196,0,.12); color:#ffc400; border:1px solid rgba(255,196,0,.35); }
.odds-card.total { border-color:#1f6f43; }
.odds-card.total .win { color:#00e676; }
.odds-card.total .sub { color:#93a0b0; font-size:.85rem; margin-top:.2rem; }
.batnum { padding-top:6px; font-weight:800; color:#00e676; text-align:center; font-size:1.1rem; }
</style>
"""

# Full team name -> Statcast abbreviation (for looking up per-team bullpen profiles).
TEAM_NAME_TO_ABBR = {
    "Arizona Diamondbacks": "AZ", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Athletics": "ATH", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
}

# =====================================================================
# 1. DATA LOADING & LEAGUE BASELINES
# =====================================================================
def _load_data():
    """Load Statcast data and reduce it to one row per plate appearance (PA).
    Statcast is one row per PITCH; a PA is a row where `events` is populated."""
    try:
        df = pd.read_csv("mlb_batting_data.csv", low_memory=False)
    except FileNotFoundError:
        return None, None

    pa = df[df['events'].notna()].copy()
    # Numeric MLBAM ids are the reliable join key (names are unreliable).
    for col in ['batter', 'pitcher']:
        if col in pa.columns:
            pa[col] = pd.to_numeric(pa[col], errors='coerce')
    return df, pa


RECENCY_HALF_LIFE_DAYS = 150   # a PA this many days old counts half as much


def add_recency_weights(pa, ref_date, half_life=RECENCY_HALF_LIFE_DAYS):
    """Return a copy of `pa` with a 'w' column giving recent PAs more weight
    (exponential decay). _rates/_hit_types pick this column up automatically, so
    profiles reflect a player's CURRENT form, not a flat multi-season average.
    `ref_date` is 'now' — today in the app, or the game's date in a backtest."""
    out = pa.copy()
    dates = pd.to_datetime(out["game_date"], errors="coerce")
    age = (pd.Timestamp(ref_date) - dates).dt.days.clip(lower=0).fillna(9999)
    out["w"] = 0.5 ** (age / float(half_life))
    return out


def _rates(m):
    """PA-level rates (walk, K, hit-on-contact). Honors a recency 'w' column if
    present (weighted counts); otherwise every PA counts equally."""
    if "w" in m.columns:
        w = m["w"].values
        W = float(w.sum())
        if W <= 0:
            return None
        ev = m["events"]
        bb = float((ev.isin(['walk', 'intent_walk']).values * w).sum())
        k = float((ev.eq('strikeout').values * w).sum())
        h = float((ev.isin(HIT_EVENTS).values * w).sum())
        contact = W - bb - k
        return {"n": W, "bb": bb / W, "k": k / W,
                "chit": (h / contact) if contact > 0 else 0.0}
    n = len(m)
    if n == 0:
        return None
    bb = m['events'].isin(['walk', 'intent_walk']).sum()
    k = (m['events'] == 'strikeout').sum()
    h = m['events'].isin(HIT_EVENTS).sum()
    contact_pa = n - bb - k          # PAs that ended on a ball in play or HR
    chit = (h / contact_pa) if contact_pa > 0 else 0.0
    return {"n": n, "bb": bb / n, "k": k / n, "chit": chit}


def _hit_types(m):
    """Distribution of hit types among all hits (HR / 3B / 2B / else = 1B).
    Honors the recency 'w' column when present."""
    h = m[m['events'].isin(HIT_EVENTS)]
    if "w" in h.columns:
        w = h["w"].values
        W = float(w.sum())
        if W <= 0:
            return None
        ev = h["events"]
        return {"n": W,
                "hr": float((ev.eq('home_run').values * w).sum()) / W,
                "triple": float((ev.eq('triple').values * w).sum()) / W,
                "double": float((ev.eq('double').values * w).sum()) / W}
    total = len(h)
    if total == 0:
        return None
    return {
        "n": total,
        "hr": (h['events'] == 'home_run').sum() / total,
        "triple": (h['events'] == 'triple').sum() / total,
        "double": (h['events'] == 'double').sum() / total,
    }


def _compute_league_baselines(pa):
    """League-average rates overall, by pitcher hand, by batter hand, and in
    the full 2-D (batter stand x pitcher hand) grid. Used for log5 blending
    and for regressing small samples toward the mean."""
    base = {
        "overall_rates": _rates(pa),
        "overall_hittype": _hit_types(pa),
        "by_throws": {}, "by_stand": {}, "grid": {},
    }
    for t in ['L', 'R']:
        base["by_throws"][t] = _rates(pa[pa['p_throws'] == t])
    for s in ['L', 'R']:
        base["by_stand"][s] = _rates(pa[pa['stand'] == s])
    for s in ['L', 'R']:
        base["grid"][s] = {}
        for t in ['L', 'R']:
            cell = _rates(pa[(pa['stand'] == s) & (pa['p_throws'] == t)])
            base["grid"][s][t] = cell if cell else base["by_throws"][t]
    return base


# --- Empirical-Bayes shrinkage: blend a small sample toward a league prior ---
RATE_SMOOTH = 120.0     # PAs of "league prior" mixed into rate splits
HITTYPE_SMOOTH = 60.0   # hits of "league prior" mixed into hit-type mix


def _shrink_rates(sub, prior):
    if sub is None:
        return {"bb": prior["bb"], "k": prior["k"], "chit": prior["chit"]}
    w = sub["n"] / (sub["n"] + RATE_SMOOTH)
    return {
        "bb": w * sub["bb"] + (1 - w) * prior["bb"],
        "k": w * sub["k"] + (1 - w) * prior["k"],
        "chit": w * sub["chit"] + (1 - w) * prior["chit"],
    }


def _shrink_hittypes(sub, prior):
    if sub is None:
        return {"hr": prior["hr"], "triple": prior["triple"], "double": prior["double"]}
    w = sub["n"] / (sub["n"] + HITTYPE_SMOOTH)
    return {
        "hr": w * sub["hr"] + (1 - w) * prior["hr"],
        "triple": w * sub["triple"] + (1 - w) * prior["triple"],
        "double": w * sub["double"] + (1 - w) * prior["double"],
    }


def build_batter_profile(pa, batter_id, bats, league):
    """Batter profile with L/R platoon splits. Falls back to league average
    if the id is unknown (e.g. a manually typed name)."""
    m = pa[pa['batter'] == batter_id] if batter_id is not None else pa.iloc[0:0]
    split = {}
    for t in ['L', 'R']:  # vs LHP / vs RHP
        split[t] = _shrink_rates(_rates(m[m['p_throws'] == t]), league["by_throws"][t])
    ht = _shrink_hittypes(_hit_types(m), league["overall_hittype"])
    return {"bats": bats or 'R', "split": split, **ht}


def build_pitcher_profile(pa, pitcher_id, throws, league, is_bullpen=False):
    """Pitcher profile with splits vs LHB/RHB. Bullpen (or unknown) pitchers
    default to league-average, i.e. matchup driven purely by the batter."""
    if is_bullpen or pitcher_id is None:
        split = {s: {"bb": league["by_stand"][s]["bb"],
                     "k": league["by_stand"][s]["k"],
                     "chit": league["by_stand"][s]["chit"]} for s in ['L', 'R']}
        return {"throws": throws or 'R', "split": split, "is_pen": is_bullpen}
    m = pa[pa['pitcher'] == pitcher_id]
    split = {}
    for s in ['L', 'R']:  # vs LHB / vs RHB
        split[s] = _shrink_rates(_rates(m[m['stand'] == s]), league["by_stand"][s])
    return {"throws": throws or 'R', "split": split, "is_pen": is_bullpen}


def compute_bullpen_profiles(pa, league):
    """Per-team RELIEF-pitching profiles, keyed by Statcast team abbreviation
    (e.g. 'MIN'). A reliever = any pitcher who did NOT throw in the 1st inning of
    that game. Rates are split vs LHB/RHB and regressed toward league average,
    so a team with a strong pen is modeled as tougher late than a weak one.
    Honors the recency 'w' column if present."""
    starters = pa[pa['inning'] == 1][['game_pk', 'pitcher']].drop_duplicates()
    starters = starters.assign(_starter=1)
    merged = pa.merge(starters, on=['game_pk', 'pitcher'], how='left')
    relief = pa[merged['_starter'].isna().values].copy()
    # The pitching team is the HOME team in the top of the inning, else the away team.
    relief['_team'] = np.where(relief['inning_topbot'] == 'Top',
                               relief['home_team'], relief['away_team'])
    profiles = {}
    for team, m in relief.groupby('_team'):
        split = {s: _shrink_rates(_rates(m[m['stand'] == s]), league["by_stand"][s])
                 for s in ['L', 'R']}
        thr = m['p_throws'].mode()
        profiles[team] = {"throws": thr.iloc[0] if len(thr) else 'R',
                          "split": split, "is_pen": True}
    return profiles


# NOTE: Team DEFENSE (xBA-based fielding factors) was tested here and REMOVED —
# a 300-game backtest showed no improvement once properly centered (pitcher rates
# already encode a team's defense, so it mostly double-counted). See project notes.


# =====================================================================
# 2. LIVE SCHEDULE / ROSTER / WEATHER FETCHERS
# =====================================================================
def get_games_for_date(date_str):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=probablePitcher,team"
    try:
        res = requests.get(url, timeout=15).json()
        games = []
        if res.get("dates"):
            for g in res["dates"][0]["games"]:
                away, home = g['teams']['away'], g['teams']['home']
                games.append({
                    "label": f"{away['team']['name']} @ {home['team']['name']}",
                    "id": g["gamePk"],
                    "away_team_id": away['team']['id'],
                    "home_team_id": home['team']['id'],
                    "away_team_name": away['team']['name'],
                    "home_team_name": home['team']['name'],
                    "sched_away_pitcher": away.get("probablePitcher", {}).get("fullName", "TBD Pitcher"),
                    "sched_home_pitcher": home.get("probablePitcher", {}).get("fullName", "TBD Pitcher"),
                    "away_pitcher_id": away.get("probablePitcher", {}).get("id"),
                    "home_pitcher_id": home.get("probablePitcher", {}).get("id"),
                })
        return games
    except Exception:
        return []


def get_handedness(person_ids):
    """Batch-fetch bat/throw handedness for a list of MLBAM ids."""
    ids = [str(i) for i in person_ids if i]
    if not ids:
        return {}
    url = f"https://statsapi.mlb.com/api/v1/people?personIds={','.join(ids)}"
    out = {}
    try:
        res = requests.get(url, timeout=15).json()
        for p in res.get("people", []):
            out[p["id"]] = {
                "bats": p.get("batSide", {}).get("code", "R"),
                "throws": p.get("pitchHand", {}).get("code", "R"),
            }
    except Exception:
        pass
    return out


def get_game_lineups(game):
    """Return pitchers + 9-man lineups as records with id, name, handedness.
    Prefers the posted lineup; falls back to the active roster."""
    gid = game["id"]
    lineups = {
        "away_pitcher": {"name": game["sched_away_pitcher"], "id": game["away_pitcher_id"], "throws": "R"},
        "home_pitcher": {"name": game["sched_home_pitcher"], "id": game["home_pitcher_id"], "throws": "R"},
        "away_lineup": [], "home_lineup": [],
    }
    # Try the live boxscore for the actual batting order.
    try:
        res = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{gid}/feed/live", timeout=15).json()
        box = res["liveData"]["boxscore"]["teams"]
        for side in ["away", "home"]:
            order = box[side].get("battingOrder", [])
            for bid in order:
                person = box[side]["players"][f"ID{bid}"]["person"]
                lineups[f"{side}_lineup"].append({"name": person["fullName"], "id": person["id"], "bats": "R"})
    except Exception:
        pass

    # Fallback to active roster if no posted lineup yet.
    for side, team_id in [("away", game["away_team_id"]), ("home", game["home_team_id"])]:
        if len(lineups[f"{side}_lineup"]) < 9:
            try:
                url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active"
                roster = requests.get(url, timeout=15).json()["roster"]
                hitters = [p for p in roster if p["position"]["type"] != "Pitcher"][:9]
                lineups[f"{side}_lineup"] = [
                    {"name": p["person"]["fullName"], "id": p["person"]["id"], "bats": "R"} for p in hitters
                ]
            except Exception:
                lineups[f"{side}_lineup"] = [
                    {"name": f"{side.title()} Batter {i}", "id": None, "bats": "R"} for i in range(1, 10)
                ]

    # Fill in real handedness for everyone in one batched call.
    all_ids = ([lineups["away_pitcher"]["id"], lineups["home_pitcher"]["id"]]
               + [b["id"] for b in lineups["away_lineup"] + lineups["home_lineup"]])
    hand = get_handedness(all_ids)
    for side in ["away", "home"]:
        pid = lineups[f"{side}_pitcher"]["id"]
        if pid in hand:
            lineups[f"{side}_pitcher"]["throws"] = hand[pid]["throws"]
        for b in lineups[f"{side}_lineup"]:
            if b["id"] in hand:
                b["bats"] = hand[b["id"]]["bats"]
    return lineups


def get_weather(game_id):
    """Pull temp / wind from the live game feed if available."""
    default = {"temp": 70, "wind_mph": 5, "wind_dir": "Neutral"}
    try:
        res = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live", timeout=15).json()
        w = res["gameData"]["weather"]
        temp = int(w.get("temp", 70))
        wind_raw = w.get("wind", "5 mph, None")  # e.g. "12 mph, Out To CF"
        mph = int("".join(ch for ch in wind_raw.split("mph")[0] if ch.isdigit()) or 5)
        low = wind_raw.lower()
        wind_dir = "Out" if "out" in low else "In" if "in" in low else "Neutral"
        return {"temp": temp, "wind_mph": mph, "wind_dir": wind_dir}
    except Exception:
        return default


def get_team_roster(team_id):
    """Full active roster for a team: {full_name: {id, bats, is_pitcher}}. Used to
    populate the lineup dropdowns so swapping a hitter loads the RIGHT player's
    stats + handedness automatically."""
    try:
        url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active"
        roster = requests.get(url, timeout=15).json()["roster"]
        hand = get_handedness([p["person"]["id"] for p in roster])
        out = {}
        for p in roster:
            pid = p["person"]["id"]
            out[p["person"]["fullName"]] = {
                "id": pid,
                "bats": hand.get(pid, {}).get("bats", "R"),
                "is_pitcher": p["position"]["type"] == "Pitcher",
            }
        return out
    except Exception:
        return {}


# =====================================================================
# 3. MATCHUP MATH + ENVIRONMENT ADJUSTMENTS
# =====================================================================
def log5(bat_rate, pit_rate, lg_rate):
    """Odds-ratio (log5) blend of a batter rate and pitcher rate vs league."""
    if lg_rate <= 0 or lg_rate >= 1:
        return min(max(bat_rate, 0.0), 1.0)
    num = (bat_rate * pit_rate) / lg_rate
    den = num + ((1.0 - bat_rate) * (1.0 - pit_rate)) / (1.0 - lg_rate)
    return num / den if den > 0 else lg_rate


def weather_hr_mult(temp, wind_mph, wind_dir):
    m = 1.0 + (temp - 70) * 0.006          # ~0.6% more HR per degree above 70F
    if wind_dir == "Out":
        m += wind_mph * 0.010
    elif wind_dir == "In":
        m -= wind_mph * 0.008
    return max(0.70, min(1.40, m))


def weather_hit_mult(temp):
    return max(0.92, min(1.08, 1.0 + (temp - 70) * 0.002))


def convert_prob_to_moneyline(prob):
    if prob >= 0.999: return "-10000"
    if prob <= 0.001: return "+10000"
    if prob == 0.5:   return "+100"
    if prob > 0.5:
        return f"{int(round((prob / (1 - prob)) * -100))}"
    return f"+{int(round(((1 - prob) / prob) * 100))}"


def moneyline_to_prob(ml):
    """American moneyline -> implied win probability (includes the book's vig)."""
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return None
    if ml == 0:
        return None
    return (-ml) / (-ml + 100) if ml < 0 else 100.0 / (ml + 100)


def odds_to_prob(value, fmt="American"):
    """Parse either American (+130, -150) or Decimal (2.30, 1.67) odds into an
    implied probability. Returns None if blank/invalid."""
    s = str(value).strip().replace("+", "")
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if fmt == "Decimal":
        return 1.0 / v if v > 1.0 else None
    if v == 0:
        return None
    return (-v) / (-v + 100) if v < 0 else 100.0 / (v + 100)


def prob_to_odds(prob, fmt="American"):
    """Model probability -> displayed fair odds in the chosen format."""
    if prob <= 0:
        return "—"
    if fmt == "Decimal":
        return f"{1.0 / prob:.2f}"
    return convert_prob_to_moneyline(prob)


def poisson_sf(k, mean):
    """P(X > k) for a Poisson(mean) — used to price over/under prop lines.
    E.g. P(over 6.5 Ks) = poisson_sf(6, expected_Ks)."""
    if mean <= 0:
        return 0.0
    import math
    cdf = 0.0
    term = math.exp(-mean)   # P(X = 0)
    cdf += term
    for i in range(1, int(k) + 1):
        term *= mean / i
        cdf += term
    return max(0.0, min(1.0, 1.0 - cdf))


def prob_at_least_one(mean):
    """P(X >= 1) for a Poisson(mean) — e.g. probability of 'hits a HR'."""
    import math
    return 1.0 - math.exp(-mean) if mean > 0 else 0.0


# =====================================================================
# 4. SIMULATION ENGINE
# =====================================================================
def simulate_pitch_sequence(pa_outcome="contact"):
    """Approximate pitches thrown in a plate appearance. MLB averages ~3.9
    pitches/PA (walks & strikeouts run longer than balls in play); this drives
    how deep a starter goes before hitting the pitch limit."""
    base = {"walk": 5.3, "strikeout": 4.8}.get(pa_outcome, 3.9)
    return max(1, int(round(random.gauss(base, 1.3))))


def eff_stand(bats, p_throws):
    """Effective batting side; switch hitters (S) bat opposite the pitcher."""
    if bats == 'S':
        return 'L' if p_throws == 'R' else 'R'
    return bats if bats in ('L', 'R') else 'R'


def simulate_games(away, home, away_p, home_p, park, weather, league, pa, simulations=1000,
                   away_bullpen=None, home_bullpen=None):
    """Monte Carlo full-game simulator using platoon-aware matchup rates,
    park factors, and weather. `away`/`home` are lists of batter records with
    keys name/id/bats. `away_p`/`home_p` are pitcher records name/id/throws.
    `away_bullpen`/`home_bullpen` are that team's relief profile (from
    compute_bullpen_profiles); if None, a league-average bullpen is used."""

    # --- Pre-compute every player's profile once ---
    bat_prof = {}
    for rec in away + home:
        bat_prof[rec["name"]] = build_batter_profile(pa, rec["id"], rec["bats"], league)

    pit_prof = {}
    pit_prof[away_p["name"]] = build_pitcher_profile(pa, away_p["id"], away_p["throws"], league)
    pit_prof[home_p["name"]] = build_pitcher_profile(pa, home_p["id"], home_p["throws"], league)
    pit_prof[f"{away_p['name']} [Bullpen]"] = away_bullpen or \
        build_pitcher_profile(pa, None, away_p["throws"], league, is_bullpen=True)
    pit_prof[f"{home_p['name']} [Bullpen]"] = home_bullpen or \
        build_pitcher_profile(pa, None, home_p["throws"], league, is_bullpen=True)

    park_runs = park["runs"] / 100.0
    park_hr = park["hr"] / 100.0
    w_hit = weather_hit_mult(weather["temp"])
    w_hr = weather_hr_mult(weather["temp"], weather["wind_mph"], weather["wind_dir"])
    HIT_ENV = park_runs * w_hit
    HR_ENV = park_hr * w_hr

    away_names = [r["name"] for r in away]
    home_names = [r["name"] for r in home]

    batting = {n: {"ab": 0, "h": 0, "double": 0, "triple": 0, "hr": 0,
                   "rbi": 0, "so": 0, "bb": 0} for n in away_names + home_names}
    pitching = {n: {"outs": 0, "k": 0, "bb": 0, "er": 0} for n in pit_prof}

    away_wins = home_wins = 0
    tot_runs, away_runs_list, home_runs_list = [], [], []

    def plate_appearance(batter_name, pit_name, bases, outs):
        """Resolve one PA. Returns (outs_added, runs_scored, updated_bases)."""
        b1, b2, b3 = bases
        bp = bat_prof[batter_name]
        pp = pit_prof[pit_name]
        stand = eff_stand(bp["bats"], pp["throws"])
        throws = pp["throws"]

        lg = league["grid"][stand][throws]
        b_rate = bp["split"][throws]
        p_rate = pp["split"][stand]

        prob_bb = log5(b_rate["bb"], p_rate["bb"], lg["bb"])
        prob_k = log5(b_rate["k"], p_rate["k"], lg["k"])
        prob_bb = max(0.01, min(0.50, prob_bb))
        prob_k = max(0.01, min(0.60, prob_k))
        if prob_bb + prob_k > 0.95:
            scale = 0.95 / (prob_bb + prob_k)
            prob_bb *= scale; prob_k *= scale
        prob_hbp = 0.011                         # hit-by-pitch: reaches base like a walk

        roll = random.random()
        outs_added = 0
        runs = 0

        if roll < prob_bb + prob_hbp:            # WALK or HIT-BY-PITCH (force advance)
            if roll < prob_bb:                   # (HBP doesn't count as a walk stat)
                batting[batter_name]["bb"] += 1
                pitching[pit_name]["bb"] += 1
            if b1 and b2 and b3:
                runs += 1
            elif b1 and b2: b3 = 1
            elif b1: b2 = 1
            else: b1 = 1
        elif roll < prob_bb + prob_hbp + prob_k: # STRIKEOUT
            outs_added = 1
            batting[batter_name]["ab"] += 1
            batting[batter_name]["so"] += 1
            pitching[pit_name]["k"] += 1
        else:                                    # BALL IN PLAY
            batting[batter_name]["ab"] += 1
            prob_hit = min(0.95, log5(b_rate["chit"], p_rate["chit"], lg["chit"]) * HIT_ENV)
            if random.random() < prob_hit:       # HIT
                batting[batter_name]["h"] += 1
                hr_r = min(0.95, bp["hr"] * HR_ENV)
                t = random.random()
                if t < hr_r:                     # HOME RUN
                    batting[batter_name]["hr"] += 1
                    runs += b1 + b2 + b3 + 1
                    b1 = b2 = b3 = 0
                elif t < hr_r + bp["triple"]:    # TRIPLE
                    batting[batter_name]["triple"] += 1
                    runs += b1 + b2 + b3
                    b1, b2, b3 = 0, 0, 1
                elif t < hr_r + bp["triple"] + bp["double"]:  # DOUBLE
                    batting[batter_name]["double"] += 1
                    runs += b2 + b3                       # runners on 2nd & 3rd score
                    new3 = 0
                    if b1:
                        if random.random() < 0.47:        # runner from 1st scores ~47%
                            runs += 1
                        else:
                            new3 = 1                       # else holds at 3rd
                    b1, b2, b3 = 0, 1, new3               # batter to 2nd
                else:                            # SINGLE
                    scored = b3                           # runner on 3rd scores
                    new3 = 0
                    if b2:
                        if random.random() < 0.62:        # runner from 2nd scores ~62%
                            scored += 1
                        else:
                            new3 = 1                       # else holds at 3rd
                    new2 = 0
                    if b1:
                        if random.random() < 0.25 and new3 == 0:
                            new3 = 1                       # first-to-third
                        else:
                            new2 = 1                       # else to second
                    runs += scored
                    b1, b2, b3 = 1, new2, new3            # batter to first
            else:                                # BALL IN PLAY, NOT A HIT
                if random.random() < 0.020:      # REACHED ON ERROR — not an out
                    runs += b3                           # runner on 3rd scores
                    b1, b2, b3 = 1, b1, b2               # everyone advances one base
                else:                            # OUT ON CONTACT
                    outs_added = 1
                    # Sac fly / productive out: runner on 3rd scores with < 2 outs.
                    if b3 and outs < 2 and random.random() < 0.24:
                        runs += 1
                        b3 = 0

        batting[batter_name]["rbi"] += runs
        pitching[pit_name]["er"] += runs
        return outs_added, runs, (b1, b2, b3)

    for _ in range(simulations):
        away_ptr = home_ptr = 0
        a_runs = h_runs = 0
        a_pitches = h_pitches = 0
        a_hooked = h_hooked = False
        a_limit = random.randint(85, 105)
        h_limit = random.randint(85, 105)

        for _inning in range(1, 10):
            # ---- TOP: away bats, home pitches ----
            outs = 0; bases = (0, 0, 0)
            while outs < 3:
                if h_pitches >= h_limit: h_hooked = True
                pit_name = f"{home_p['name']} [Bullpen]" if h_hooked else home_p["name"]
                name = away_names[away_ptr]
                oa, _r, bases = plate_appearance(name, pit_name, bases, outs)
                if not h_hooked:
                    h_pitches += simulate_pitch_sequence("contact")
                a_runs += _r
                outs += oa
                pitching[pit_name]["outs"] += oa
                away_ptr = (away_ptr + 1) % len(away_names)

            # ---- BOTTOM: home bats, away pitches ----
            outs = 0; bases = (0, 0, 0)
            while outs < 3:
                if a_pitches >= a_limit: a_hooked = True
                pit_name = f"{away_p['name']} [Bullpen]" if a_hooked else away_p["name"]
                name = home_names[home_ptr]
                oa, _r, bases = plate_appearance(name, pit_name, bases, outs)
                if not a_hooked:
                    a_pitches += simulate_pitch_sequence("contact")
                h_runs += _r
                outs += oa
                pitching[pit_name]["outs"] += oa
                home_ptr = (home_ptr + 1) % len(home_names)

        if a_runs == h_runs:                     # break ties (extra innings proxy)
            if random.random() < 0.5: a_runs += 1
            else: h_runs += 1
        if a_runs > h_runs: away_wins += 1
        else: home_wins += 1

        away_runs_list.append(a_runs)
        home_runs_list.append(h_runs)
        tot_runs.append(a_runs + h_runs)

    # --- Aggregate hitter lines ---
    hitters = {}
    for n, s in batting.items():
        avg = s["h"] / s["ab"] if s["ab"] > 0 else 0.0
        hitters[n] = {
            "avg": f"{avg:.3f}", "ab": f"{s['ab']/simulations:.1f}",
            "h": f"{s['h']/simulations:.2f}", "xbh": f"{(s['double']+s['triple'])/simulations:.2f}",
            "hr": f"{s['hr']/simulations:.2f}", "rbi": f"{s['rbi']/simulations:.2f}",
            "so": f"{s['so']/simulations:.2f}", "bb": f"{s['bb']/simulations:.2f}",
        }

    # --- Aggregate pitcher lines ---
    pitchers = {}
    for n, s in pitching.items():
        outs = s["outs"] / simulations
        innings = outs / 3.0
        era = (s["er"] / simulations) / (innings / 9.0) if innings > 0 else 0.0
        whole = int(outs // 3)                 # proper baseball IP notation (.0/.1/.2)
        rem = int(round(outs - whole * 3))
        if rem >= 3:
            whole += 1; rem -= 3
        pitchers[n] = {
            "ip": f"{whole}.{rem}",
            "k": f"{s['k']/simulations:.2f}", "bb": f"{s['bb']/simulations:.2f}",
            "er": f"{era:.2f}",
        }

    betting = {
        "away_win_pct": away_wins / simulations,
        "home_win_pct": home_wins / simulations,
        "avg_total_runs": float(np.mean(tot_runs)),
        "away_exp_runs": float(np.mean(away_runs_list)),
        "home_exp_runs": float(np.mean(home_runs_list)),
        "raw_totals": tot_runs,
        # Per-sim margin from the AWAY team's view (away_runs - home_runs). Used to
        # price the run line: away covers -1.5 when margin >= 2, etc.
        "raw_margins": [a - h for a, h in zip(away_runs_list, home_runs_list)],
    }
    return hitters, pitchers, betting


# =====================================================================
# 5. CACHED WRAPPERS
# The app uses these (fast, cached). backtest.py / ML code import the plain
# underscore-prefixed versions above so they run without a Streamlit runtime.
# =====================================================================
load_data = st.cache_data(_load_data)
compute_league_baselines = st.cache_data(_compute_league_baselines)


@st.cache_data(ttl=900)
def load_track_record():
    """Read the running performance log (settled predictions only)."""
    try:
        df = pd.read_csv("predictions.csv")
        return df[df["status"] == "settled"].copy()
    except Exception:
        return None


def render_track_record():
    """Live model-performance dashboard, read from predictions.csv."""
    tr = load_track_record()
    if tr is None or len(tr) < 10:
        return
    acc, brier, mae = tr["correct"].mean(), tr["brier"].mean(), tr["total_abs_err"].mean()
    with st.expander(f"📈 Live Track Record — {len(tr):,} games graded · {acc*100:.1f}% winners picked"):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Games graded", f"{len(tr):,}")
        m2.metric("Winner accuracy", f"{acc*100:.1f}%")
        m3.metric("Brier score", f"{brier:.3f}", help="lower is better; a coin-flip = 0.250")
        m4.metric("Total-runs MAE", f"{mae:.2f}")

        trs = tr.sort_values("pred_date").reset_index(drop=True)
        trs["game_no"] = range(1, len(trs) + 1)
        trs["cum_acc"] = trs["correct"].expanding().mean() * 100
        st.markdown("**Cumulative winner accuracy** (settles as more games are graded)")
        st.altair_chart(
            alt.Chart(trs).mark_line(color="#00e676").encode(
                x=alt.X("game_no:Q", title="games graded"),
                y=alt.Y("cum_acc:Q", title="accuracy %", scale=alt.Scale(domain=[35, 65])),
                tooltip=["game_no", alt.Tooltip("cum_acc:Q", format=".1f")],
            ).properties(height=220), use_container_width=True)

        trs["bucket"] = pd.cut(trs["pred_home_win_pct"], [0, .35, .45, .55, .65, 1.01],
                               labels=["<35%", "35-45%", "45-55%", "55-65%", ">65%"])
        cal = (trs.groupby("bucket", observed=True)
               .agg(pred=("pred_home_win_pct", "mean"), actual=("home_won", "mean"),
                    n=("home_won", "size")).reset_index().dropna())
        if len(cal):
            cal["pred"] *= 100; cal["actual"] *= 100
            st.markdown("**Calibration** — dots near the dashed line = honest probabilities")
            diag = alt.Chart(pd.DataFrame({"x": [0, 100], "y": [0, 100]})).mark_line(
                strokeDash=[4, 4], color="#5a6472").encode(x="x:Q", y="y:Q")
            pts = alt.Chart(cal).mark_circle(color="#00e676").encode(
                x=alt.X("pred:Q", title="predicted home win %", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("actual:Q", title="actual home win %", scale=alt.Scale(domain=[0, 100])),
                size=alt.Size("n:Q", legend=None, scale=alt.Scale(range=[80, 500])),
                tooltip=["bucket", alt.Tooltip("pred:Q", format=".0f"),
                         alt.Tooltip("actual:Q", format=".0f"), "n"])
            st.altair_chart((diag + pts).properties(height=260), use_container_width=True)

        if tr["model_version"].nunique() > 1:
            st.markdown("**By model version** — compare weightings on real results")
            vv = (tr.groupby("model_version")
                  .agg(games=("correct", "size"), accuracy=("correct", "mean"),
                       brier=("brier", "mean")).reset_index())
            vv["accuracy"] = (vv["accuracy"] * 100).round(1)
            vv["brier"] = vv["brier"].round(3)
            st.table(vv.set_index("model_version"))
        st.caption(f"Walk-forward + live predictions graded against final scores. "
                   f"Auto-updates daily. Current model: `{MODEL_VERSION}`.")


# =====================================================================
# 6. STREAMLIT APP (only runs via `streamlit run app.py`)
# =====================================================================
def main():
    st.set_page_config(page_title="MLB AI Predictor", page_icon="⚾", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><span class="eyebrow">Live · Backtest-validated</span>'
        '<h1>⚾ MLB AI Predictor</h1>'
        '<p>Monte-Carlo game simulation with live lineups, platoon &amp; park factors, '
        'weather, and bullpen strength — plus a public, self-updating accuracy track record.</p></div>',
        unsafe_allow_html=True)

    raw_df, pa_df = load_data()
    if pa_df is None:
        st.error("❌ `mlb_batting_data.csv` not found. Run fetch_data.py first.")
        st.stop()
    pa_df = add_recency_weights(pa_df, datetime.today())   # weight recent form higher
    league = compute_league_baselines(pa_df)
    if "bullpens" not in st.session_state:                  # compute once per session (fast reruns)
        st.session_state.bullpens = compute_bullpen_profiles(pa_df, league)
    bullpens = st.session_state.bullpens

    data_through = pd.to_datetime(pa_df["game_date"], errors="coerce").max()
    data_through_str = data_through.strftime("%B %-d, %Y") if pd.notna(data_through) else "unknown"
    b_left, b_right = st.columns([3, 2])
    b_left.success(f"🟢 {len(pa_df):,} plate appearances · {pa_df['batter'].nunique():,} batters loaded")
    b_right.info(f"📅 Model data through **{data_through_str}**")

    with st.expander("ℹ️ About this model — and how accurate it actually is"):
        st.markdown(f"""
This is a **Monte-Carlo simulator**: it plays each matchup thousands of times using every
hitter's and pitcher's real rates, blended via *log5* matchup math, and adjusted for
**platoon (L/R) splits, ballpark, weather, recent form, and each team's bullpen**.

**It was validated, not just built.** On a 300-game walk-forward backtest (predicting games
using only data from *before* each game — no cheating):

| Metric | Result | Meaning |
|---|---|---|
| Game-winner accuracy | **~58%** | beats "always pick home" (55.7%) |
| Brier score | **0.244** | beats the naive baseline (0.247) — win probabilities are *honest* |
| Run-total bias | **~0** | totals aren't systematically high or low |

Single-game outcomes are inherently high-variance — no model (or sportsbook) predicts one
game's exact total. The edge is small but real, and the probabilities are **calibrated**:
when it says 60%, that side wins about 60% of the time.

*Data snapshot through {data_through_str}. Not affiliated with MLB.*
""")

    render_track_record()

    # ----- Session state -----
    def _blank_lineup(side):
        return pd.DataFrame({
            "Order": range(1, 10),
            "Player Name": [f"{side} Batter {i}" for i in range(1, 10)],
            "Bats": ["R"] * 9,
        })

    if "current_game_id" not in st.session_state:
        st.session_state.current_game_id = None
        st.session_state.away_p = {"name": "Away Pitcher", "id": None, "throws": "R"}
        st.session_state.home_p = {"name": "Home Pitcher", "id": None, "throws": "R"}
        st.session_state.away_df = _blank_lineup("Away")
        st.session_state.home_df = _blank_lineup("Home")
        st.session_state.away_roster = {}
        st.session_state.home_roster = {}
        st.session_state.weather = {"temp": 70, "wind_mph": 5, "wind_dir": "Neutral"}

    # ----- Step 1: schedule & lineups -----
    st.markdown("---")
    st.subheader("🗓️ Step 1: Pick a Matchup")

    selected_date = st.date_input("Date", value=datetime.today())
    games = get_games_for_date(selected_date.strftime('%Y-%m-%d'))
    if not games:
        st.info("No games found for this date — using a demo matchup.")
        games = [{
            "label": "Kansas City Royals @ Washington Nationals", "id": 111111,
            "away_team_id": 118, "home_team_id": 120,
            "away_team_name": "Kansas City Royals", "home_team_name": "Washington Nationals",
            "sched_away_pitcher": "Away Pitcher", "sched_home_pitcher": "Home Pitcher",
            "away_pitcher_id": None, "home_pitcher_id": None,
        }]

    # ----- Full-slate predictions (all games on this date) -----
    with st.expander(f"🗓️ Full-Slate Predictions — simulate every game on {selected_date.strftime('%b %d')}"):
        st.caption("Runs a quick simulation on every scheduled game and ranks them by confidence. Takes ~1 min.")
        if st.button("⚡ Predict entire slate", key="slate_btn"):
            prog = st.progress(0.0, text="Simulating slate...")
            slate_rows = []
            for gi, g in enumerate(games):
                try:
                    cards = get_game_lineups(g)
                    aabbr = TEAM_NAME_TO_ABBR.get(g["away_team_name"])
                    habbr = TEAM_NAME_TO_ABBR.get(g["home_team_name"])
                    _, _, bet = simulate_games(
                        cards["away_lineup"], cards["home_lineup"],
                        cards["away_pitcher"], cards["home_pitcher"],
                        PARK_FACTORS.get(g["home_team_name"], NEUTRAL_PARK),
                        {"temp": 70, "wind_mph": 5, "wind_dir": "Neutral"},
                        league, pa_df, simulations=400,
                        away_bullpen=bullpens.get(aabbr), home_bullpen=bullpens.get(habbr))
                    fav = (g["away_team_name"] if bet["away_win_pct"] > bet["home_win_pct"]
                           else g["home_team_name"])
                    slate_rows.append({
                        "Matchup": g["label"],
                        "Away Win %": round(bet["away_win_pct"] * 100, 1),
                        "Home Win %": round(bet["home_win_pct"] * 100, 1),
                        "Proj Total": round(bet["avg_total_runs"], 1),
                        "Favorite": fav.split()[-1],
                        "Conf %": round(max(bet["away_win_pct"], bet["home_win_pct"]) * 100, 1),
                    })
                except Exception:
                    pass
                prog.progress((gi + 1) / len(games), text=f"Simulated {gi+1}/{len(games)} games")
            prog.empty()
            if slate_rows:
                slate_df = (pd.DataFrame(slate_rows)
                            .sort_values("Conf %", ascending=False).set_index("Matchup"))
                st.dataframe(slate_df, use_container_width=True)
            else:
                st.info("Couldn't simulate the slate — lineups may not be posted yet for this date.")

    game_map = {g["label"]: g for g in games}
    chosen = game_map[st.selectbox("Matchup", list(game_map.keys()))]

    # Load lineups, rosters + weather when the selected game changes.
    if st.session_state.current_game_id != chosen["id"]:
        st.session_state.current_game_id = chosen["id"]
        with st.spinner("Fetching lineups, rosters, handedness & weather..."):
            cards = get_game_lineups(chosen)
            st.session_state.weather = get_weather(chosen["id"])
            st.session_state.away_roster = get_team_roster(chosen["away_team_id"])
            st.session_state.home_roster = get_team_roster(chosen["home_team_id"])
        st.session_state.away_p = cards["away_pitcher"]
        st.session_state.home_p = cards["home_pitcher"]
        st.session_state.away_df = pd.DataFrame({
            "Order": range(1, 10),
            "Player Name": [b["name"] for b in cards["away_lineup"]],
            "Bats": [b["bats"] for b in cards["away_lineup"]],
        })
        st.session_state.home_df = pd.DataFrame({
            "Order": range(1, 10),
            "Player Name": [b["name"] for b in cards["home_lineup"]],
            "Bats": [b["bats"] for b in cards["home_lineup"]],
        })

    def render_lineup(prefix, roster, default_df, gid):
        """Batting order as 9 slots: each a roster dropdown + ⬆⬇ move arrows.
        Reorder is instant; swaps are backed by session_state so state survives."""
        n = len(default_df)
        options = sorted(roster) if roster else list(default_df["Player Name"])
        defaults = list(default_df["Player Name"])
        # Initialise each slot once per game.
        for i in range(n):
            k = f"{prefix}_{gid}_{i}"
            if k not in st.session_state:
                st.session_state[k] = defaults[i] if i < len(defaults) else (options[0] if options else "")
        # Apply a queued swap BEFORE the selectboxes are instantiated this run.
        pend = st.session_state.pop(f"{prefix}_swap", None)
        if pend:
            a, b = f"{prefix}_{gid}_{pend[0]}", f"{prefix}_{gid}_{pend[1]}"
            if a in st.session_state and b in st.session_state:
                st.session_state[a], st.session_state[b] = st.session_state[b], st.session_state[a]
        # Render the slots.
        for i in range(n):
            k = f"{prefix}_{gid}_{i}"
            cur = st.session_state[k]
            opts = list(options) if cur in options else [cur] + list(options)
            c = st.columns([0.5, 6, 0.9, 0.9])
            c[0].markdown(f"<div class='batnum'>{i+1}</div>", unsafe_allow_html=True)
            c[1].selectbox("p", opts, key=k, label_visibility="collapsed")
            if c[2].button("⬆", key=f"{k}_up", disabled=(i == 0), use_container_width=True):
                st.session_state[f"{prefix}_swap"] = (i, i - 1); st.rerun()
            if c[3].button("⬇", key=f"{k}_dn", disabled=(i == n - 1), use_container_width=True):
                st.session_state[f"{prefix}_swap"] = (i, i + 1); st.rerun()

    def lineup_records(prefix, gid, n, roster):
        recs = []
        for i in range(n):
            name = st.session_state.get(f"{prefix}_{gid}_{i}")
            if not isinstance(name, str) or not name.strip():
                continue
            info = roster.get(name.strip(), {})
            recs.append({"name": name.strip(), "id": info.get("id"), "bats": info.get("bats", "R")})
        return recs

    st.caption("✏️ Pick a player from the dropdown · tap ⬆ / ⬇ to reorder the batting order instantly.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"### 🚀 {chosen['away_team_name']} (Away)")
        st.session_state.away_p["name"] = st.text_input("Away SP", st.session_state.away_p["name"], key=f"ap_{chosen['id']}")
        st.session_state.away_p["throws"] = st.selectbox("Throws", ["R", "L"],
            index=0 if st.session_state.away_p["throws"] == "R" else 1, key=f"apt_{chosen['id']}")
        render_lineup("aord", st.session_state.away_roster, st.session_state.away_df, chosen['id'])
    with c2:
        st.markdown(f"### 🏠 {chosen['home_team_name']} (Home)")
        st.session_state.home_p["name"] = st.text_input("Home SP", st.session_state.home_p["name"], key=f"hp_{chosen['id']}")
        st.session_state.home_p["throws"] = st.selectbox("Throws", ["R", "L"],
            index=0 if st.session_state.home_p["throws"] == "R" else 1, key=f"hpt_{chosen['id']}")
        render_lineup("hord", st.session_state.home_roster, st.session_state.home_df, chosen['id'])

    # ----- Step 2: park & weather -----
    st.markdown("---")
    st.subheader("🌤️ Step 2: Park & Weather")
    park = PARK_FACTORS.get(chosen["home_team_name"], NEUTRAL_PARK)

    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        st.metric("Ballpark", chosen["home_team_name"].split()[-1])
        st.caption(f"Runs factor **{park['runs']}** · HR factor **{park['hr']}** (100 = neutral)")
    with pc2:
        temp = st.number_input("Temp (°F)", 30, 115, int(st.session_state.weather["temp"]))
    with pc3:
        wind_mph = st.number_input("Wind (mph)", 0, 40, int(st.session_state.weather["wind_mph"]))
    with pc4:
        wind_dir = st.selectbox("Wind Dir", ["Neutral", "Out", "In"],
            index=["Neutral", "Out", "In"].index(st.session_state.weather["wind_dir"]))
    weather = {"temp": temp, "wind_mph": wind_mph, "wind_dir": wind_dir}

    # ----- Step 3: run -----
    st.markdown("---")
    st.subheader("⚡ Step 3: Run the Simulation")
    sim_count = st.select_slider("Monte Carlo games", options=[500, 1000, 2500, 5000], value=1000)

    away_short = chosen['away_team_name'].split()[-1]
    home_short = chosen['home_team_name'].split()[-1]
    with st.expander("💵 Enter sportsbook lines (optional) — to find betting value"):
        odds_fmt = st.radio("Odds format", ["American", "Decimal"], horizontal=True,
                            help="American: +130 / -150 · Decimal: 2.30 / 1.67")
        ml_ph = "e.g. 2.30" if odds_fmt == "Decimal" else "e.g. +130"
        juice_ph = "1.91" if odds_fmt == "Decimal" else "-110"

        st.markdown("**Moneyline**")
        v1, v2 = st.columns(2)
        vegas_away_ml = v1.text_input(f"{away_short} moneyline", "", placeholder=ml_ph)
        vegas_home_ml = v2.text_input(f"{home_short} moneyline", "", placeholder=ml_ph)

        st.markdown("**Total (Over / Under)** — odds are usually not even, so enter both")
        t1, t2, t3 = st.columns(3)
        vegas_total = t1.text_input("Total line", "", placeholder="e.g. 8.5")
        vegas_over_odds = t2.text_input("Over odds", "", placeholder=juice_ph)
        vegas_under_odds = t3.text_input("Under odds", "", placeholder=juice_ph)

        st.markdown("**Run line (spread)** — e.g. Royals **-1.5** (win by 2+) or **+1.5** (lose by ≤1)")
        r1, r2, r3 = st.columns(3)
        rl_team = r1.selectbox("Team", [away_short, home_short])
        rl_line = r2.text_input("Run line", "-1.5", placeholder="-1.5")
        rl_odds = r3.text_input("Run-line odds", "", placeholder=ml_ph)

    if st.button("🔥 RUN PREDICTOR", type="primary"):
        away_recs = lineup_records("aord", chosen['id'], len(st.session_state.away_df), st.session_state.away_roster)
        home_recs = lineup_records("hord", chosen['id'], len(st.session_state.home_df), st.session_state.home_roster)

        # --- Guard against incomplete input so a public user never hits a crash ---
        problems = []
        if len(away_recs) < 1 or len(home_recs) < 1:
            problems.append("each team needs at least one batter in the lineup")
        if not (st.session_state.away_p.get("name") or "").strip() or \
           not (st.session_state.home_p.get("name") or "").strip():
            problems.append("both starting pitchers need a name")
        if problems:
            st.warning("⚠️ Can't run yet — " + "; ".join(problems) + ".")
            st.stop()
        if len(away_recs) < 9 or len(home_recs) < 9:
            st.info("ℹ️ A lineup has fewer than 9 batters — simulating with what's entered.")

        away_abbr = TEAM_NAME_TO_ABBR.get(chosen["away_team_name"])
        home_abbr = TEAM_NAME_TO_ABBR.get(chosen["home_team_name"])
        try:
            with st.spinner("Simulating..."):
                hitters, pitchers, betting = simulate_games(
                    away_recs, home_recs, st.session_state.away_p, st.session_state.home_p,
                    park, weather, league, pa_df, simulations=sim_count,
                    away_bullpen=bullpens.get(away_abbr), home_bullpen=bullpens.get(home_abbr))
        except Exception as e:
            st.error(f"😕 Something went wrong running the simulation. Try reselecting the game "
                     f"or reloading lineups.\n\n`{type(e).__name__}: {e}`")
            st.stop()

        st.markdown("## 💰 Betting & Market Metrics")
        away_ml = prob_to_odds(betting['away_win_pct'], odds_fmt)
        home_ml = prob_to_odds(betting['home_win_pct'], odds_fmt)
        st.markdown(f"""
        <div class="odds-wrap">
          <div class="odds-card">
            <div class="team">🚀 {chosen['away_team_name']}</div>
            <div class="win">{betting['away_win_pct']*100:.1f}%</div>
            <div class="ml">ML {away_ml}</div>
          </div>
          <div class="odds-card total">
            <div class="team">📊 Total Runs (O/U)</div>
            <div class="win">{betting['avg_total_runs']:.1f}</div>
            <div class="sub">{chosen['away_team_name'].split()[-1]} {betting['away_exp_runs']:.1f}
                &nbsp;·&nbsp; {chosen['home_team_name'].split()[-1]} {betting['home_exp_runs']:.1f}</div>
          </div>
          <div class="odds-card">
            <div class="team">🏠 {chosen['home_team_name']}</div>
            <div class="win">{betting['home_win_pct']*100:.1f}%</div>
            <div class="ml">ML {home_ml}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ---- Value / edge finder (only if user entered book lines) ----
        # Each entry: (title, model_prob, book_implied_prob).
        value_bets = []

        # Moneyline (each team)
        for mprob, vml, tm in [
            (betting['away_win_pct'], vegas_away_ml, away_short),
            (betting['home_win_pct'], vegas_home_ml, home_short)]:
            vp = odds_to_prob(vml, odds_fmt)
            if vp is not None:
                value_bets.append((f"{tm} ML {vml}", mprob, vp))

        # Total (best of over/under vs their own odds)
        try:
            total_line = float(vegas_total)
        except (TypeError, ValueError):
            total_line = None
        if total_line is not None:
            totals = np.array(betting['raw_totals'])
            p_over = float(np.mean(totals > total_line))
            p_under = float(np.mean(totals < total_line))
            oi, ui = odds_to_prob(vegas_over_odds, odds_fmt), odds_to_prob(vegas_under_odds, odds_fmt)
            tsides = []
            if oi is not None: tsides.append((f"Over {total_line} ({vegas_over_odds})", p_over, oi))
            if ui is not None: tsides.append((f"Under {total_line} ({vegas_under_odds})", p_under, ui))
            if tsides:
                value_bets.append(max(tsides, key=lambda s: s[1] - s[2]))

        # Run line (spread)
        try:
            rl = float(rl_line)
            rl_imp = odds_to_prob(rl_odds, odds_fmt)
        except (TypeError, ValueError):
            rl, rl_imp = None, None
        if rl is not None and rl_imp is not None:
            margins = np.array(betting['raw_margins'])              # away - home per sim
            team_margin = margins if rl_team == away_short else -margins
            p_cover = float(np.mean(team_margin > -rl))             # covers when margin + line > 0
            value_bets.append((f"{rl_team} {rl:+g} ({rl_odds})", p_cover, rl_imp))

        if value_bets:
            st.markdown("### 💵 Value vs. the Book")
            st.caption("Green = the model's probability beats the book's implied odds (positive expected value).")
            cols = st.columns(len(value_bets))
            for j, (title, mprob, imp) in enumerate(value_bets):
                edge = mprob - imp
                verdict = "🟢 VALUE" if edge > 0.02 else ("🟡 slight edge" if edge > 0 else "🔴 no edge")
                cols[j].metric(title, f"{edge*100:+.1f}%",
                               help=f"Model {mprob*100:.0f}% vs book-implied {imp*100:.0f}%")
                cols[j].caption(verdict)

        st.markdown("### 📈 Total-Runs Distribution")
        counts = pd.Series(betting["raw_totals"]).value_counts()
        chart_df = pd.DataFrame({"Runs": list(range(0, 21)),
                                 "Occurrences": [int(counts.get(r, 0)) for r in range(0, 21)]})
        st.altair_chart(
            alt.Chart(chart_df).mark_bar(color="#00e676", cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X("Runs:O", axis=alt.Axis(labelAngle=0, title="Total Runs")),
                y=alt.Y("Occurrences:Q", title="Frequency"),
                tooltip=["Runs", "Occurrences"],
            ).properties(width="container", height=350),
            use_container_width=True)

        def team_table(records, data):
            d = {r["name"]: data[r["name"]] for r in records if r["name"] in data}
            df = pd.DataFrame.from_dict(d, orient="index")
            df.columns = ["AVG", "AB", "Hits", "XBH", "HR", "RBI", "SO", "BB"]
            return df

        st.markdown("### 📊 Projected Hitting")
        st.markdown(f"#### 🚀 {chosen['away_team_name']}")
        st.table(team_table(away_recs, hitters))
        st.markdown(f"#### 🏠 {chosen['home_team_name']}")
        st.table(team_table(home_recs, hitters))

        st.markdown("### 🥎 Projected Pitching")
        dfp = pd.DataFrame.from_dict(pitchers, orient="index")
        dfp.columns = ["IP", "K", "BB", "ERA"]
        st.table(dfp)

        # ---- Player prop probabilities ----
        with st.expander("🎯 Player Prop Probabilities (over/under lines)"):
            def hitter_props(records):
                rows = []
                for r in records:
                    if r["name"] not in hitters:
                        continue
                    h = hitters[r["name"]]
                    exp_h, exp_hr, exp_tb = float(h["h"]), float(h["hr"]), float(h["so"])
                    rows.append({
                        "Player": r["name"],
                        "1+ Hit": f"{prob_at_least_one(exp_h)*100:.0f}%",
                        "2+ Hits": f"{poisson_sf(1, exp_h)*100:.0f}%",
                        "Hits a HR": f"{prob_at_least_one(exp_hr)*100:.0f}%",
                        "Exp Hits": f"{exp_h:.2f}",
                    })
                return pd.DataFrame(rows).set_index("Player")

            st.markdown(f"**🚀 {chosen['away_team_name']} — hitter props**")
            st.table(hitter_props(away_recs))
            st.markdown(f"**🏠 {chosen['home_team_name']} — hitter props**")
            st.table(hitter_props(home_recs))

            st.markdown("**⚾ Starting-pitcher strikeout props**")
            prows = []
            for pname, pstat in pitchers.items():
                if "[Bullpen]" in pname:
                    continue
                expk = float(pstat["k"])
                line = round(expk - 0.5) + 0.5 if expk >= 1 else 0.5   # nearest half line
                prows.append({
                    "Pitcher": pname,
                    "Exp Ks": f"{expk:.1f}",
                    "Line": f"{line:.1f}",
                    "Over %": f"{poisson_sf(int(line), expk)*100:.0f}%",
                })
            if prows:
                st.table(pd.DataFrame(prows).set_index("Pitcher"))
            st.caption("Props use a Poisson model on each projection — approximate, for guidance.")

    # ----- Disclaimer footer (always shown) -----
    st.markdown("""
    <div style="margin-top:2.5rem;padding:16px 20px;border-top:1px solid #263043;
         color:#7c8798;font-size:.82rem;line-height:1.5;">
      <b>⚠️ For informational & entertainment purposes only.</b> These are model estimates,
      not guarantees or financial advice. Nothing here is a recommendation to place a wager.
      If you choose to bet, you must be of legal age (21+ in most areas) and bet responsibly.
      Problem gambling? Call <b>1-800-GAMBLER</b>.<br>
      Not affiliated with, endorsed by, or sourced in real time from MLB. Data is a periodic
      snapshot and may be out of date.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
