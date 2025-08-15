# fpl_auto_bot.py
"""
Autonomous FPL bot using PL_SESSION only.
- Uses official FPL endpoints with session cookie.
- Builds squad, sets captain/vice, performs free transfers.
- Logs everything.
- No email/password required.
"""

import os
import json
import traceback
from collections import defaultdict
from datetime import datetime, timezone

import requests
import pandas as pd
from bs4 import BeautifulSoup

# -------------------------
# Env
# -------------------------
print("✅ Bot starting…")
PL_SESSION = os.getenv("PL_SESSION")
TEAM_ID = os.getenv("FPL_TEAM_ID")
assert PL_SESSION, "Missing PL_SESSION env"
assert TEAM_ID, "Missing FPL_TEAM_ID env"
TEAM_ID = int(str(TEAM_ID).strip())

BASE = "https://fantasy.premierleague.com"
BOOTSTRAP_URL = f"{BASE}/api/bootstrap-static/"
MY_TEAM_URL = f"{BASE}/api/my-team/{TEAM_ID}/"
TRANSFERS_URL = f"{BASE}/api/my-team/{TEAM_ID}/transfers/"
PICKS_URL = f"{BASE}/api/my-team/{TEAM_ID}/"

MAX_PER_CLUB = 3
BUDGET_TENTHS = 1000  # £100.0m
SQUAD_SHAPE = {1: 2, 2: 5, 3: 5, 4: 3}  # Total 15
XI_MINIMA = {1: 1, 2: 3, 3: 2, 4: 1}
POSITION_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}

# -------------------------
# Utils
# -------------------------
def utc_now():
    return datetime.now(timezone.utc)

def hours_to_deadline(iso: str) -> float:
    dl = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (dl - utc_now()).total_seconds() / 3600.0

def session_with_cookie():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Referer": BASE,
    })
    s.cookies.set("pl_session", PL_SESSION, domain=".fantasy.premierleague.com")
    return s

def fetch_bootstrap():
    print("[FETCH] bootstrap-static…")
    r = requests.get(BOOTSTRAP_URL, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_my_team(session: requests.Session):
    print("[FETCH] my-team…")
    r = session.get(MY_TEAM_URL, timeout=30)
    if r.status_code == 403:
        raise RuntimeError("403 Forbidden: PL_SESSION invalid or expired.")
    r.raise_for_status()
    return r.json()

# -------------------------
# Player processing
# -------------------------
def build_player_table(bs: dict) -> pd.DataFrame:
    els = pd.DataFrame(bs['elements'])
    teams = pd.DataFrame(bs['teams']).rename(columns={"id":"team_id","name":"team_name"})
    els = els.merge(teams[["team_id","team_name"]], left_on="team", right_on="team_id", how="left")
    els['ep_next_f'] = els['ep_next'].fillna(0.0).astype(float)
    els['form_f'] = els['form'].fillna(0.0).astype(float)
    els['ict_f'] = els['ict_index'].fillna(0.0).astype(float)
    els['now_cost_i'] = els['now_cost'].astype(int)
    injury = els['news'].fillna('').str.contains("injury|doubt|knock|illness|suspend", case=False, regex=True)
    els['avail_mult'] = 0.6
    els.loc[(els['status']=='a') & (~injury), 'avail_mult'] = 1.0
    els.loc[injury, 'avail_mult'] = 0.3
    return els

def score_players(els: pd.DataFrame) -> pd.DataFrame:
    base = (els['ep_next_f']*10 + els['form_f']*2 + els['ict_f']*0.8) * els['avail_mult']
    pos_mult = els['element_type'].map({1:0.9, 2:1.05, 3:1.1, 4:1.0})
    out = els.copy()
    out['score'] = base * pos_mult
    return out

def pick_squad(els_scored: pd.DataFrame, budget_tenths: int) -> list[int]:
    selected = []
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
    xi, pos_ct = [], defaultdict(int)
    for pos, mn in XI_MINIMA.items():
        pool = squad[squad['element_type']==pos].head(mn)
        xi.extend(pool['id'].astype(int).tolist())
        pos_ct[pos] += len(pool)
    rem = 11 - len(xi)
    taken = set(xi)
    for _, row in squad.iterrows():
        if rem == 0:
            break
        pid = int(row['id'])
        if pid in taken: continue
        p = int(row['element_type'])
        if p == 1 and pos_ct[p]>=1: continue
        if p == 4 and pos_ct[p]>=3: continue
        xi.append(pid)
        pos_ct[p] += 1
        rem -=1
    xi_df = squad[squad['id'].isin(xi)].sort_values('score', ascending=False)
    captain = int(xi_df.iloc[0]['id'])
    vice = int(xi_df.iloc[1]['id'])
    return xi, captain, vice

# -------------------------
# Posting
# -------------------------
def post_transfers(session: requests.Session, payload: dict, note: str):
    headers = {"Referer": BASE, "Content-Type": "application/json"}
    print(f"[POST] Transfers ({note}) → {TRANSFERS_URL}")
    r = session.post(TRANSFERS_URL, data=json.dumps(payload), headers=headers, timeout=30)
    print("[POST] Status:", r.status_code)
    r.raise_for_status()

def post_picks(session: requests.Session, picks_payload: dict):
    headers = {"Referer": BASE, "Content-Type": "application/json"}
    print(f"[POST] Picks → {PICKS_URL}")
    r = session.post(PICKS_URL, data=json.dumps(picks_payload), headers=headers, timeout=30)
    print("[POST] Status:", r.status_code)

# -------------------------
# Main routine
# -------------------------
def weekly_routine():
    print("===== FPL AUTO RUN =====")
    bs = fetch_bootstrap()
    next_ev = next((ev for ev in bs.get('events', []) if ev.get('is_next')), None)
    if not next_ev:
        print("[INFO] No upcoming GW. Exit."); return
    gw = int(next_ev['id']); deadline_iso = next_ev['deadline_time']
    hrs = hours_to_deadline(deadline_iso)
    print(f"Next GW: {gw} | Deadline: {deadline_iso} | Hours to deadline: {hrs:.2f}")

    session = session_with_cookie()
    my = fetch_my_team(session)
    bank = int(my['transfers']['bank']); current_picks = [int(p['element']) for p in my.get('picks',[])]
    print(f"[TEAM] Bank: {bank/10:.1f}m | Owned count: {len(current_picks)}")

    players = build_player_table(bs)
    scored = score_players(players)

    # GW1 unlimited
    if gw==1 and hrs>0:
        print("[GW1] Building optimal 15…")
        ideal = pick_squad(scored, BUDGET_TENTHS)
        xi, cap, vice = pick_xi(scored, ideal)
        payload = {"entry":TEAM_ID,"event":gw,"transfers":[],"chip":None,"squad":ideal,"captain":cap,"vice_captain":vice,"confirmed":True}
        post_transfers(session, payload, note="GW1 squad set")
        picks_payload = {"picks":[{"element":pid,"position":idx+1,"is_captain":pid==cap,"is_vice_captain":pid==vice} for idx,pid in enumerate(xi)],"chips":[],"entry_history":{"event":gw}}
        post_picks(session, picks_payload)
        print("[GW1] Squad & picks applied."); return

    # GW2+ within 24h
    if 0<hrs<=24:
        print("[GW] <24h to deadline → planning free transfers…")
        target = pick_squad(scored, BUDGET_TENTHS)
        desired = []; owned_set=set(current_picks)
        for pid in target:
            if pid in owned_set: desired.append(pid)
        for pid in target:
            if len(desired)>=15: break
            if pid not in desired: desired.append(pid)
        free_left=int(my['transfers'].get('limit',1))-int(my['transfers'].get('made',0))
        print(f"[GW] Free transfers available: {free_left}")
        outs, ins=[], []
        if free_left>0:
            for pid in current_picks:
                if pid not in desired and len(outs)<free_left: outs.append(pid)
            for pid in desired:
                if pid not in owned_set and len(ins)<free_left: ins.append(pid)
        transfers=[]
        if len(outs)==len(ins) and len(ins)>0:
            for o,i in zip(outs,ins):
                transfers.append({"element_out":o,"element_in":i,"purchase_price":0,"selling_price":0})
        xi_source = desired if len(desired)==15 else current_picks
        xi, cap, vice = pick_xi(scored, xi_source)
        payload = {"entry":TEAM_ID,"event":gw,"transfers":transfers,"chip":None,"confirmed":True,"captain":cap,"vice_captain":vice}
        if transfers: post_transfers(session,payload,note="GW free transfers")
        else:
            picks_payload = {"picks":[{"element":pid,"position":idx+1,"is_captain":pid==cap,"is_vice_captain":pid==vice} for idx,pid in enumerate(xi)],"chips":[],"entry_history":{"event":gw}}
            post_picks(session,picks_payload)
        print("[GW] Weekly action complete."); return

    print("[INFO] Not within action window. Exit.")

if __name__=="__main__":
    try: weekly_routine()
    except Exception as e:
        print("[ERROR]", e)
        traceback.print_exc()
