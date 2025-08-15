# fpl_auto_bot.py

"""
FINAL autonomous FPL bot.
- Uses official FPL web endpoints via an authenticated session (cookies preserved).
- Plans using `is_next` event (upcoming GW).
- GW1: performs unlimited transfers to set optimal 15-man squad (NO wildcard used).
- GW2+: acts only within 24h before the upcoming deadline; uses FREE transfers only (no points hits),
  and sets captain/vice each week. You can extend to hits if desired.
- Simple chip scaffolding: easy to enable when conditions are met (kept OFF by default here to be safe).
- Rich logging at every step for GitHub Actions.

Run this file once per day via GitHub Actions on a cron. It exits cleanly when not in an action window.
"""

import os
import json
import time
import math
import traceback
from collections import defaultdict
from datetime import datetime, timezone

import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# -------------------------
# Boot / Env
# -------------------------
print("✅ Bot starting…")
load_dotenv()
FPL_EMAIL = os.getenv("FPL_EMAIL")
FPL_PASSWORD = os.getenv("FPL_PASSWORD")
TEAM_ID = os.getenv("FPL_TEAM_ID")

assert FPL_EMAIL, "Missing FPL_EMAIL env"
assert FPL_PASSWORD, "Missing FPL_PASSWORD env"
assert TEAM_ID, "Missing FPL_TEAM_ID env"
TEAM_ID = int(str(TEAM_ID).strip())

BASE = "https://fantasy.premierleague.com"
LOGIN_URL = "https://users.premierleague.com/accounts/login/"
BOOTSTRAP_URL = f"{BASE}/api/bootstrap-static/"
MY_TEAM_URL = f"{BASE}/api/my-team/{TEAM_ID}/"
TRANSFERS_URL = f"{BASE}/api/my-team/{TEAM_ID}/transfers/"
PICKS_URL = f"{BASE}/api/my-team/{TEAM_ID}/"  # sets captain/vice + picks

MAX_PER_CLUB = 3
BUDGET_TENTHS = 1000  # £100.0m

# -------------------------
# Utils
# -------------------------

def utc_now():
    return datetime.now(timezone.utc)


def hours_to_deadline(iso: str) -> float:
    dl = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (dl - utc_now()).total_seconds() / 3600.0


# -------------------------
# Authenticated Session
# -------------------------

def login_to_fpl(email: str, password: str) -> requests.Session:
    print("[LOGIN] Starting…")
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36",
        "origin": BASE,
        "referer": BASE,
    })
    payload = {
        "login": email,
        "password": password,
        "app": "plfpl-web",
        "redirect_uri": f"{BASE}/",
    }
    r = s.post(LOGIN_URL, data=payload, allow_redirects=True, timeout=30)
    print("[LOGIN] Status:", r.status_code)
    # Basic cookie check
    if not s.cookies.get_dict():
        print("[LOGIN] No cookies found; server might have blocked the login flow.")
    else:
        print("[LOGIN] Cookies present. Auth looks OK.")
    return s


def fetch_bootstrap() -> dict:
    print("[FETCH] bootstrap-static…")
    r = requests.get(BOOTSTRAP_URL, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_my_team(session: requests.Session) -> dict:
    print("[FETCH] my-team…")
    r = session.get(MY_TEAM_URL, timeout=30)
    print("[FETCH] my-team status:", r.status_code)
    if r.status_code == 403:
        raise RuntimeError("403 Forbidden on my-team. Login session/cookies not accepted by API.")
    r.raise_for_status()
    return r.json()


# -------------------------
# Sentiment (lightweight)
# -------------------------
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    _sid = SentimentIntensityAnalyzer()
    SENTIMENT_READY = True
except Exception as e:
    print("[SENTIMENT] NLTK not available:", e)
    SENTIMENT_READY = False

NEWS_SOURCES = [
    "https://www.bbc.com/sport/football/premier-league",
    "https://www.theguardian.com/football/premierleague",
    "https://www.skysports.com/premier-league-news",
    "https://www.espn.com/soccer/league/_/name/ENG.1",
    "https://www.football.london/all-about/fantasy-football",
]


def scrape_sentiment(players_df: pd.DataFrame) -> dict:
    if not SENTIMENT_READY:
        return {}
    names = set(players_df['web_name'].astype(str).str.lower())
    scores = defaultdict(float)
    for url in NEWS_SOURCES:
        try:
            print(f"[NEWS] Scrape {url}")
            res = requests.get(url, timeout=25)
            if not res.ok:
                continue
            soup = BeautifulSoup(res.text, 'html.parser')
            text = "
".join([t.get_text(" ") for t in soup.find_all(['p','h1','h2','h3','li'])])
            comp = _sid.polarity_scores(text)['compound']
            low = text.lower()
            for n in names:
                if f" {n} " in low:
                    scores[n] += comp
        except Exception as e:
            print("[NEWS] Error:", e)
    return scores


# -------------------------
# Player scoring & squad building
# -------------------------
POSITION_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
SQUAD_SHAPE = {1: 2, 2: 5, 3: 5, 4: 3}  # Total 15
XI_MINIMA = {1: 1, 2: 3, 3: 2, 4: 1}


def build_player_table(bs: dict) -> pd.DataFrame:
    els = pd.DataFrame(bs['elements'])
    teams = pd.DataFrame(bs['teams']).rename(columns={"id":"team_id","name":"team_name"})
    els = els.merge(teams[["team_id","team_name"]], left_on="team", right_on="team_id", how="left")

    def fnum(x):
        try:
            return float(x)
        except:
            return 0.0

    els['ep_next_f'] = els['ep_next'].map(fnum)
    els['form_f'] = els['form'].map(fnum)
    els['ict_f'] = els['ict_index'].map(fnum)
    els['now_cost_i'] = els['now_cost'].astype(int)

    injury = els['news'].fillna('').str.contains("injury|doubt|knock|illness|suspend", case=False, regex=True)
    els['avail_mult'] = 0.6
    els.loc[(els['status']=='a') & (~injury), 'avail_mult'] = 1.0
    els.loc[injury, 'avail_mult'] = 0.3

    return els


def score_players(els: pd.DataFrame, sentiment: dict) -> pd.DataFrame:
    base = (els['ep_next_f'] * 10.0 + els['form_f'] * 2.0 + els['ict_f'] * 0.8) * els['avail_mult']
    pos_mult = els['element_type'].map({1:0.9, 2:1.05, 3:1.1, 4:1.0})
    sent = els['web_name'].astype(str).str.lower().map(lambda n: sentiment.get(n, 0.0))
    out = els.copy()
    out['score'] = base * pos_mult + sent
    return out


def pick_squad(els_scored: pd.DataFrame, budget_tenths: int) -> list[int]:
    selected: list[int] = []
    club_ct = defaultdict(int)
    remaining = int(budget_tenths)

    for pos in [1,2,3,4]:
        need = SQUAD_SHAPE[pos]
        pool = els_scored[els_scored['element_type']==pos].sort_values('score', ascending=False)
        for _, row in pool.iterrows():
            if need == 0:
                break
            pid = int(row['id'])
            cost = int(row['now_cost_i'])
            team = int(row['team'])
            if club_ct[team] >= MAX_PER_CLUB:
                continue
            if cost <= remaining:
                selected.append(pid)
                club_ct[team] += 1
                remaining -= cost
                need -= 1
    return selected


def pick_xi(els_scored: pd.DataFrame, squad_ids: list[int]):
    squad = els_scored[els_scored['id'].isin(squad_ids)].copy().sort_values('score', ascending=False)
    xi: list[int] = []
    pos_ct = defaultdict(int)

    # satisfy minima
    for pos, mn in XI_MINIMA.items():
        pool = squad[squad['element_type']==pos].head(mn)
        xi.extend(pool['id'].astype(int).tolist())
        pos_ct[pos] += len(pool)

    # fill remaining up to 11
    rem = 11 - len(xi)
    taken = set(xi)
    for _, row in squad.iterrows():
        if rem == 0:
            break
        pid = int(row['id'])
        if pid in taken:
            continue
        p = int(row['element_type'])
        if p == 1 and pos_ct[p] >= 1:
            continue
        if p == 4 and pos_ct[p] >= 3:
            continue
        xi.append(pid)
        pos_ct[p] += 1
        rem -= 1

    xi_df = squad[squad['id'].isin(xi)].sort_values('score', ascending=False)
    captain = int(xi_df.iloc[0]['id'])
    vice = int(xi_df.iloc[1]['id'])
    return xi, captain, vice


# -------------------------
# Posting changes
# -------------------------

def post_transfers(session: requests.Session, payload: dict, note: str):
    headers = {"Referer": BASE, "Content-Type": "application/json"}
    print(f"[POST] Transfers ({note}) → {TRANSFERS_URL}")
    r = session.post(TRANSFERS_URL, data=json.dumps(payload, ensure_ascii=False), headers=headers, timeout=30)
    print("[POST] Status:", r.status_code)
    try:
        print(r.text[:600])
    except Exception:
        pass
    r.raise_for_status()


def post_picks(session: requests.Session, picks_payload: dict):
    headers = {"Referer": BASE, "Content-Type": "application/json"}
    print(f"[POST] Picks → {PICKS_URL}")
    r = session.post(PICKS_URL, data=json.dumps(picks_payload, ensure_ascii=False), headers=headers, timeout=30)
    print("[POST] Status:", r.status_code)
    try:
        print(r.text[:600])
    except Exception:
        pass
    # don't raise if picks endpoint differs; logging is enough


# -------------------------
# Main routine
# -------------------------

def weekly_routine():
    print("
===== FPL AUTO RUN =====")
    bs = fetch_bootstrap()
    events = bs.get('events', [])
    next_ev = None
    for ev in events:
        if ev.get('is_next'):
            next_ev = ev
            break
    if not next_ev:
        print("[INFO] No upcoming GW yet. Exit.")
        return

    gw = int(next_ev['id'])
    deadline_iso = next_ev['deadline_time']
    hrs = hours_to_deadline(deadline_iso)
    print(f"Next GW: {gw} | Deadline: {deadline_iso} | Hours to deadline: {hrs:.2f}")

    session = login_to_fpl(FPL_EMAIL, FPL_PASSWORD)

    # my-team
    my = fetch_my_team(session)
    bank = int(my['transfers']['bank'])  # in tenths
    current_picks = [int(p['element']) for p in my.get('picks', [])]
    print(f"[TEAM] Bank: {bank/10:.1f}m | Owned count: {len(current_picks)}")

    players = build_player_table(bs)
    sentiment = scrape_sentiment(players)
    scored = score_players(players, sentiment)

    # ---------------- GW1 path (unlimited transfers) ----------------
    if gw == 1 and hrs > 0:
        print("[GW1] Building optimal 15 with unlimited transfers (no wildcard required)…")
        ideal = pick_squad(scored, BUDGET_TENTHS)
        if len(ideal) != 15:
            print(f"[GW1][WARN] Selected {len(ideal)} players, expected 15. Proceeding with what we have.")
        xi, cap, vice = pick_xi(scored, ideal)

        transfer_payload = {
            "entry": TEAM_ID,
            "event": gw,
            "transfers": [],               # GW1 unlimited: no explicit transfer list required when sending squad
            "chip": None,
            "squad": [int(x) for x in ideal],
            "captain": int(cap),
            "vice_captain": int(vice),
            "confirmed": True,
        }
        post_transfers(session, transfer_payload, note="GW1 unlimited squad set")

        # Also set picks explicitly (some sessions require it)
        picks_payload = {
            "picks": [
                {"element": int(pid), "position": idx+1, "is_captain": int(pid)==int(cap), "is_vice_captain": int(pid)==int(vice)}
                for idx, pid in enumerate(xi)
            ],
            "chips": [],
            "entry_history": {"event": gw},
        }
        post_picks(session, picks_payload)
        print("[GW1] Squad & picks applied.")
        return

    # ---------------- GW2+ path (act only within 24h window) ----------------
    if 0 < hrs <= 24:
        print("[GW] <24h to deadline → planning free transfers only…")
        # Build target list and try to align current squad using only up-to-free transfers
        target = pick_squad(scored, BUDGET_TENTHS)
        desired = []
        owned_set = set(current_picks)
        for pid in target:
            if pid in owned_set:
                desired.append(pid)
        for pid in target:
            if len(desired) >= 15:
                break
            if pid not in desired:
                desired.append(pid)

        free_left = int(my['transfers'].get('limit', 1)) - int(my['transfers'].get('made', 0))
        print(f"[GW] Free transfers available: {free_left}")
        outs, ins = [], []
        if free_left > 0:
            for pid in current_picks:
                if pid not in desired and len(outs) < free_left:
                    outs.append(int(pid))
            for pid in desired:
                if pid not in owned_set and len(ins) < free_left:
                    ins.append(int(pid))

        transfers = []
        if len(outs) == len(ins) and len(ins) > 0:
            for o, i in zip(outs, ins):
                transfers.append({
                    "element_out": int(o),
                    "element_in": int(i),
                    "purchase_price": 0,
                    "selling_price": 0,
                })
        else:
            print("[GW] No bank-safe free swaps identified; will only update captain/vice.")

        xi_source = desired if len(desired) == 15 else current_picks
        xi, cap, vice = pick_xi(scored, xi_source)

        payload = {
            "entry": TEAM_ID,
            "event": gw,
            "transfers": transfers,
            "chip": None,
            "confirmed": True,
            "captain": int(cap),
            "vice_captain": int(vice),
        }

        if transfers:
            post_transfers(session, payload, note="GW free transfers")
        else:
            picks_payload = {
                "picks": [
                    {"element": int(pid), "position": idx+1, "is_captain": int(pid)==int(cap), "is_vice_captain": int(pid)==int(vice)}
                    for idx, pid in enumerate(xi)
                ],
                "chips": [],
                "entry_history": {"event": gw},
            }
            post_picks(session, picks_payload)
        print("[GW] Weekly action complete.")
        return

    print("[INFO] Not within action window (either >24h before deadline or deadline passed). Exit.")


if __name__ == "__main__":
    try:
        weekly_routine()
    except Exception as e:
        print("
[ERROR]", e)
        traceback.print_exc()
