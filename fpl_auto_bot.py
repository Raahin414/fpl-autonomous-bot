# fpl_auto_bot.py

import os
import time
import json
import math
import requests
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
from dotenv import load_dotenv

print("✅ Bot starting…")

# -------------------------
# Env / Constants
# -------------------------
load_dotenv()
EMAIL = os.getenv("FPL_EMAIL")
PASSWORD = os.getenv("FPL_PASSWORD")
TEAM_ID = os.getenv("FPL_TEAM_ID")

assert EMAIL and PASSWORD and TEAM_ID, "Missing FPL_EMAIL / FPL_PASSWORD / FPL_TEAM_ID env vars"

BASE = "https://fantasy.premierleague.com"
LOGIN_URL = f"https://users.premierleague.com/accounts/login/"
BOOTSTRAP_URL = f"{BASE}/api/bootstrap-static/"
MY_TEAM_URL = f"{BASE}/api/my-team/{TEAM_ID}/"
ENTRY_URL = f"{BASE}/api/entry/{TEAM_ID}/"
TRANSFERS_URL = f"{BASE}/api/my-team/{TEAM_ID}/transfers/"
PICKS_URL = f"{BASE}/api/my-team/{TEAM_ID}/"  # PUT picks (captain/vice)

MAX_PER_CLUB = 3
BUDGET_UNITS = 10  # costs & bank are in tenths

# -------------------------
# Helpers
# -------------------------

def session_login() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36",
        "origin": BASE,
        "referer": BASE,
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
    })
    payload = {
        "login": EMAIL,
        "password": PASSWORD,
        "app": "plfpl-web",
        "redirect_uri": f"{BASE}/",
    }
    r = s.post(LOGIN_URL, data=payload, allow_redirects=True, timeout=30)
    if r.status_code not in (200, 302):
        print("[LOGIN] Non-200:", r.status_code, r.text[:300])
    else:
        print("[LOGIN] OK")
    return s


def fetch_bootstrap(retries: int = 3) -> dict:
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(BOOTSTRAP_URL, timeout=30)
            if r.ok:
                return r.json()
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(2)
    raise RuntimeError(f"Failed to fetch bootstrap-static: {last_err}")


def get_next_gw(events: list) -> dict | None:
    for ev in events:
        if ev.get("is_next"):
            return ev
    return None


def utc_now():
    return datetime.now(timezone.utc)


def hours_to_deadline(deadline_iso: str) -> float:
    dl = datetime.strptime(deadline_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (dl - utc_now()).total_seconds() / 3600.0


# -------------------------
# Sentiment (lightweight)
# -------------------------
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon', quiet=True)
_sid = SentimentIntensityAnalyzer()

NEWS_SOURCES = [
    "https://www.bbc.com/sport/football/premier-league",
    "https://www.theguardian.com/football/premierleague",
    "https://www.skysports.com/premier-league-news",
    "https://www.espn.com/soccer/league/_/name/ENG.1",
    "https://www.football.london/all-about/fantasy-football",
]

def scrape_sentiment_by_player(players_df: pd.DataFrame) -> dict:
    names = set(players_df['web_name'].str.lower().tolist())
    scores = defaultdict(float)
    for url in NEWS_SOURCES:
        try:
            res = requests.get(url, timeout=20)
            if not res.ok:
                continue
            soup = BeautifulSoup(res.text, 'html.parser')
            texts = " ".join(p.get_text() for p in paragraphs)
            comp = _sid.polarity_scores(texts)['compound']
            # naive boost: distribute to all names mentioned
            for n in names:
                if f" {n} " in texts.lower():
                    scores[n] += comp
        except Exception as e:
            print(f"[NEWS] {url} -> {e}")
    return scores


# -------------------------
# Team building / scoring
# -------------------------
POSITION_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
SQUAD_SHAPE = {1: 2, 2: 5, 3: 5, 4: 3}  # 15-man squad counts
XI_MIN = {1: 1, 2: 3, 3: 2, 4: 1}       # starting XI constraints


def build_player_table(bs: dict) -> pd.DataFrame:
    els = pd.DataFrame(bs['elements'])
    teams = pd.DataFrame(bs['teams']).rename(columns={"id":"team_id","name":"team_name"})
    els = els.merge(teams[["team_id","team_name"]], left_on="team", right_on="team_id", how="left")

    def safe_float(x):
        try:
            return float(x)
        except:
            return 0.0

    # numeric features
    els['ep_next_f'] = els['ep_next'].apply(safe_float)
    els['form_f'] = els['form'].apply(safe_float)
    els['ict_index_f'] = els['ict_index'].apply(safe_float)
    els['now_cost_int'] = els['now_cost'].astype(int)

    # availability
    injury_pen = els['news'].fillna('').str.contains("injury|doubt|knock|illness|suspended", case=False, regex=True)
    els['avail_mult'] = 0.6
    els.loc[~injury_pen & (els['status'] == 'a'), 'avail_mult'] = 1.0
    els.loc[injury_pen, 'avail_mult'] = 0.3

    return els


def score_players(els: pd.DataFrame, sentiment: dict) -> pd.DataFrame:
    # base score
    base = (
        els['ep_next_f'] * 10.0 +
        els['form_f'] * 2.0 +
        els['ict_index_f'] * 0.8
    ) * els['avail_mult']

    # position multipliers (DEF/MID favored slightly for value)
    pos_mult = els['element_type'].map({1:0.9, 2:1.05, 3:1.1, 4:1.0})

    # sentiment boost
    sent = els['web_name'].str.lower().map(lambda n: sentiment.get(n, 0.0))

    els = els.copy()
    els['score'] = base * pos_mult + sent
    return els


def pick_squad(els_scored: pd.DataFrame, budget: int) -> list:
    """Greedy by score while respecting budget, club limit, and squad shape."""
    selected = []
    club_counts = defaultdict(int)
    remaining = budget

    for pos in [1,2,3,4]:
        need = SQUAD_SHAPE[pos]
        pool = els_scored[els_scored['element_type']==pos].sort_values('score', ascending=False)
        for _, row in pool.iterrows():
            if need == 0:
                break
            if club_counts[row['team']] >= MAX_PER_CLUB:
                continue
            cost = int(row['now_cost_int'])
            if cost <= remaining:
                selected.append(int(row['id']))
                club_counts[row['team']] += 1
                remaining -= cost
                need -= 1
    return selected


def pick_starting_xi(els_scored: pd.DataFrame, squad_ids: list) -> tuple[list,int,int]:
    """Return (xi_ids, captain_id, vice_id). Ensure valid formation."""
    squad = els_scored[els_scored['id'].isin(squad_ids)].copy()
    squad = squad.sort_values('score', ascending=False)

    # Start with best by position respecting XI minima
    xi = []
    pos_counts = defaultdict(int)

    # First, satisfy minima
    for pos, min_need in XI_MIN.items():
        pool = squad[squad['element_type']==pos].head(min_need)
        xi.extend(pool['id'].astype(int).tolist())
        pos_counts[pos] += len(pool)

    # Fill remaining slots to 11 with best remaining players
    remaining_slots = 11 - len(xi)
    already = set(xi)
    for _, row in squad.iterrows():
        if remaining_slots == 0:
            break
        pid = int(row['id'])
        if pid in already:
            continue
        # keep plausible formation (max 3 FWD, at least 3 DEF etc) — simple guard
        p = int(row['element_type'])
        if p == 1 and pos_counts[p] >= 1:
            continue  # only 1 GK in XI
        if p == 4 and pos_counts[p] >= 3:
            continue
        xi.append(pid)
        pos_counts[p] += 1
        remaining_slots -= 1

    # Captain = top score in XI, vice = second
    xi_df = squad[squad['id'].isin(xi)].sort_values('score', ascending=False)
    captain = int(xi_df.iloc[0]['id'])
    vice = int(xi_df.iloc[1]['id'])
    return xi, captain, vice


# -------------------------
# Current team & bank
# -------------------------

def fetch_my_team(session: requests.Session) -> dict:
    r = session.get(MY_TEAM_URL, timeout=30)
    r.raise_for_status()
    return r.json()


# -------------------------
# Apply changes (GW1 unlimited vs weekly transfers)
# -------------------------

def post_transfers(session: requests.Session, payload: dict) -> None:
    headers = {"Referer": BASE, "Content-Type": "application/json"}
    r = session.post(TRANSFERS_URL, data=json.dumps(payload), headers=headers, timeout=30)
    print("[TRANSFERS]", r.status_code)
    try:
        print(r.text[:500])
    except Exception:
        pass


def set_picks(session: requests.Session, picks_payload: dict) -> None:
    headers = {"Referer": BASE, "Content-Type": "application/json"}
    r = session.post(PICKS_URL, data=json.dumps(picks_payload), headers=headers, timeout=30)
    print("[PICKS]", r.status_code)
    try:
        print(r.text[:300])
    except Exception:
        pass


# -------------------------
# Main routine
# -------------------------

def weekly_routine():
    print("
===== FPL AUTO RUN =====")
    bs = fetch_bootstrap()
    events = bs.get('events', [])
    next_ev = get_next_gw(events)
    if not next_ev:
        print("No next GW found yet. Exiting.")
        return

    gw_id = int(next_ev['id'])
    deadline = next_ev['deadline_time']
    hours_left = hours_to_deadline(deadline)
    print(f"Next GW: {gw_id} | Deadline: {deadline} | Hours to deadline: {hours_left:.2f}")

    session = session_login()
    my = fetch_my_team(session)
    bank = int(my['transfers']['bank'])  # in tenths
    current_picks = [int(p['element']) for p in my['picks']]
    current_set = set(current_picks)

    players = build_player_table(bs)
    sentiment = scrape_sentiment_by_player(players)
    scored = score_players(players, sentiment)

    # Budget: bank + current squad value (we don't know exact SV; use my['transfers']['value'])
    total_value = int(my['transfers']['value'])  # in tenths
    available_budget = bank  # for replacement deltas we will stay <= bank for safety

    # --- GW1: Unlimited transfers before deadline ---
    if gw_id == 1 and hours_left > 0:
        print("GW1 window open – building optimal 15 with unlimited transfers…")
        ideal_squad = pick_squad(scored, budget=1000)  # 100.0 budget
        if len(ideal_squad) != 15:
            print("[WARN] Could not fill 15 players under constraints. Selected:", len(ideal_squad))
        xi, cap, vice = pick_starting_xi(scored, ideal_squad)

        payload = {
            "entry": int(TEAM_ID),
            "event": gw_id,
            "transfers": [],
            "wildcard": True,           # unlimited-like action
            "chip": None,
            "squad": [int(x) for x in ideal_squad],
            "captain": int(cap),
            "vice_captain": int(vice),
            "confirmed": True,
        }
        post_transfers(session, payload)
        print("GW1 squad applied.")
        return

    # --- GW>1: Within 24h to deadline, make limited transfers (use free only) ---
    if hours_left <= 24 and hours_left > 0:
        print("<24h to deadline – planning free transfers only.")
        # Build an ideal squad to compare against
        target_squad = pick_squad(scored, budget=1000)  # reference list by score
        target_set = set(target_squad)

        # Keep currently owned that are also in top target
        desired = []
        for pid in target_squad:
            if pid in current_set:
                desired.append(pid)
        # Add best remaining to reach 15
        for pid in target_squad:
            if len(desired) >= 15:
                break
            if pid not in desired:
                desired.append(pid)

        # Plan up to free transfers (no hits)
        free_left = int(my['transfers'].get('limit', 1)) - int(my['transfers'].get('made', 0))
        print(f"Free transfers available: {free_left}")
        outs, ins = [], []
        if free_left > 0:
            for pid in current_picks:
                if pid not in desired and len(outs) < free_left:
                    outs.append(pid)
            for pid in desired:
                if pid not in current_set and len(ins) < free_left:
                    ins.append(pid)

        transfers = []
        # simple bank-safe: only proceed if we find same count and assume price neutral (server enforces exact prices)
        if len(outs) == len(ins) and len(ins) > 0:
            for o, i in zip(outs, ins):
                transfers.append({
                    "element_out": int(o),
                    "element_in": int(i),
                    "purchase_price": 0,
                    "selling_price": 0,
                })
        else:
            print("No bank-safe free transfers found. Skipping.")

        # Recompute XI on the post-transfer desired set (fallback to current if empty)
        squad_for_xi = desired if len(desired)==15 else current_picks
        xi, cap, vice = pick_starting_xi(scored, squad_for_xi)

        payload = {
            "entry": int(TEAM_ID),
            "event": gw_id,
            "transfers": transfers,
            "chip": None,
            "confirmed": True,
            "captain": int(cap),
            "vice_captain": int(vice),
        }
        if transfers:
            post_transfers(session, payload)
        else:
            print("Nothing to transfer – only setting captain/vice if needed.")
            # send picks update payload
            picks_payload = {
                "picks": [{"element": int(pid), "position": idx+1, "is_captain": int(pid)==int(cap), "is_vice_captain": int(pid)==int(vice)} for idx, pid in enumerate(xi)],
                "chips": [],
                "entry_history": {"event": gw_id},
            }
            set_picks(session, picks_payload)
        print("Weekly update done.")
        return

    print("Not within action window. Exiting.")


if __name__ == "__main__":
    weekly_routine()


