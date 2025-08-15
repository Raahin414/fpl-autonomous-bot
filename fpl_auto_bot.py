import os
import requests
import json
from datetime import datetime, timezone

# === CONFIG FROM GITHUB SECRETS ===
FPL_EMAIL = os.environ["FPL_EMAIL"]
FPL_PASSWORD = os.environ["FPL_PASSWORD"]
TEAM_ID = os.environ["FPL_TEAM_ID"]

BASE_URL = "https://fantasy.premierleague.com/api"

# === LOGIN AND SESSION ===
def fpl_login():
    print("[LOGIN] Starting...")
    session = requests.Session()
    payload = {
        "login": FPL_EMAIL,
        "password": FPL_PASSWORD,
        "app": "plfpl-web",
        "redirect_uri": "https://fantasy.premierleague.com/"
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://fantasy.premierleague.com/"
    }
    res = session.post("https://users.premierleague.com/accounts/login/", data=payload, headers=headers)
    if res.status_code == 200 and "csrftoken" in session.cookies:
        print("[LOGIN] Success.")
    else:
        raise RuntimeError(f"[LOGIN] Failed with status {res.status_code}")
    return session

# === FETCH FUNCTIONS ===
def get_my_team(session):
    print("[FETCH] My team...")
    url = f"{BASE_URL}/my-team/{TEAM_ID}/"
    r = session.get(url)
    if r.status_code == 403:
        raise RuntimeError("403 Forbidden on my-team. Login failed.")
    r.raise_for_status()
    return r.json()

def get_bootstrap(session):
    print("[FETCH] bootstrap-static...")
    r = session.get(f"{BASE_URL}/bootstrap-static/")
    r.raise_for_status()
    return r.json()

def get_transfers_history(session):
    print("[FETCH] transfer history...")
    r = session.get(f"{BASE_URL}/entry/{TEAM_ID}/transfers/")
    r.raise_for_status()
    return r.json()

# === ACTION FUNCTIONS ===
def make_transfers(session, transfers):
    if not transfers:
        print("[ACTION] No transfers required.")
        return
    print(f"[ACTION] Making {len(transfers)} transfer(s)...")
    url = f"{BASE_URL}/transfers/{TEAM_ID}/"
    payload = {
        "confirmed": True,
        "entry": int(TEAM_ID),
        "transfers": transfers,
    }
    headers = {"Content-Type": "application/json"}
    r = session.post(url, headers=headers, data=json.dumps(payload))
    r.raise_for_status()
    print("[ACTION] Transfers complete:", r.json())

def play_chip(session, chip_name, gw_id):
    print(f"[ACTION] Playing chip: {chip_name}...")
    url = f"{BASE_URL}/entry/{TEAM_ID}/chip-play/"
    payload = {"chip": chip_name, "event": gw_id}
    r = session.post(url, json=payload)
    r.raise_for_status()
    print(f"[ACTION] Chip '{chip_name}' played successfully.")

# === DECISION LOGIC ===
def decide_transfers(my_team, bootstrap, gw_id):
    """
    Original transfer logic:
    - GW1 → unlimited transfers, pick best 15 players
    - Later → 1 FT/week, pick best upgrade
    """
    transfers = []
    team_ids = [p["element"] for p in my_team["picks"]]
    players = bootstrap["elements"]

    if gw_id == 1:
        print("[LOGIC] GW1 - Unlimited transfers logic.")
        best_players = sorted(players, key=lambda x: (float(x["form"]), -x["now_cost"]), reverse=True)[:15]
        transfers = []
        for old, new in zip(team_ids, [p["id"] for p in best_players]):
            if old != new:
                transfers.append({
                    "element_in": new,
                    "element_out": old,
                    "purchase_price": next(p["now_cost"] for p in players if p["id"] == new),
                    "selling_price": next(p["now_cost"] for p in players if p["id"] == old)
                })
    else:
        print(f"[LOGIC] GW{gw_id} - Limited transfers logic.")
        lowest_value_player = None
        for pid in team_ids:
            p = next(pl for pl in players if pl["id"] == pid)
            if not lowest_value_player or p["now_cost"] < lowest_value_player["now_cost"]:
                lowest_value_player = p

        available_players = [pl for pl in players if pl["id"] not in team_ids]
        best_player = max(available_players, key=lambda x: float(x["form"]))

        if lowest_value_player and best_player:
            transfers.append({
                "element_in": best_player["id"],
                "element_out": lowest_value_player["id"],
                "purchase_price": best_player["now_cost"],
                "selling_price": lowest_value_player["now_cost"]
            })
            print(f"[LOGIC] Transfer OUT: {lowest_value_player['web_name']} → IN: {best_player['web_name']}")

    return transfers

def decide_chip(gw_info, transfers_history):
    """
    Original chip logic:
    - GW1 → Triple Captain if unused
    - Later → Bench Boost GW19 if unused
    """
    chips_used = {t.get("chip") for t in transfers_history if t.get("chip")}
    if gw_info["id"] == 1 and "3xc" not in chips_used:
        return "3xc"
    if gw_info["id"] == 19 and "bboost" not in chips_used:
        return "bboost"
    return None

# === MAIN ROUTINE ===
def weekly_routine():
    print("===== FPL AUTO RUN =====")
    session = fpl_login()

    bootstrap = get_bootstrap(session)
    gw_info = next(gw for gw in bootstrap["events"] if gw["is_next"] or gw["is_current"])
    deadline = datetime.fromisoformat(gw_info["deadline_time"].replace("Z", "+00:00"))
    hours_to_deadline = (deadline - datetime.now(timezone.utc)).total_seconds() / 3600
    print(f"[INFO] GW: {gw_info['id']} | Deadline: {deadline} | Hours left: {hours_to_deadline:.2f}")

    my_team = get_my_team(session)
    transfers_history = get_transfers_history(session)

    # Decide and make transfers
    transfers = decide_transfers(my_team, bootstrap, gw_info["id"])
    make_transfers(session, transfers)

    # Decide and play chip
    chip_to_play = decide_chip(gw_info, transfers_history)
    if chip_to_play:
        play_chip(session, chip_to_play, gw_info["id"])

    print("[DONE] Weekly routine completed.")

if __name__ == "__main__":
    weekly_routine()
