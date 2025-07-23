# fpl_auto_bot.py

import os
import time
import requests
import pandas as pd
import numpy as np
import schedule
import datetime
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
from openpyxl import Workbook

print("\U0001F4E6 Downloading NLTK resources...")
import nltk
nltk.download('vader_lexicon')

print("\u2705 Bot started execution.")

# Load environment variables
load_dotenv()
EMAIL = os.getenv("FPL_EMAIL")
PASSWORD = os.getenv("FPL_PASSWORD")
TEAM_ID = os.getenv("FPL_TEAM_ID")

# FPL API endpoints
LOGIN_URL = "https://users.premierleague.com/accounts/login/"
TRANSFERS_URL = f"https://fantasy.premierleague.com/api/my-team/{TEAM_ID}/transfers/"
MY_TEAM_URL = f"https://fantasy.premierleague.com/api/my-team/{TEAM_ID}/"
ENTRY_URL = f"https://fantasy.premierleague.com/api/entry/{TEAM_ID}/"

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Fetch data from FPL
def fetch_fpl_data():
    return requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()

# Get current gameweek and deadline
def get_current_gw_and_deadline():
    events = fetch_fpl_data()["events"]
    for event in events:
        if event["is_current"]:
            deadline = datetime.datetime.strptime(event["deadline_time"], "%Y-%m-%dT%H:%M:%SZ")
            return event["id"], deadline
    return None, None

# Analyze sentiment from top 5 football news sources
def analyze_sentiment():
    sources = [
        "https://www.bbc.com/sport/football/premier-league",
        "https://www.theguardian.com/football/premierleague",
        "https://www.espn.com/soccer/team/_/id/359/english-premier-league",
        "https://www.skysports.com/premier-league-news",
        "https://www.football.london/all-about/fantasy-football"
    ]
    sentiment_scores = {}
    for url in sources:
        try:
            res = requests.get(url, timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                text = p.get_text()
                score = sid.polarity_scores(text)['compound']
                for word in text.split():
                    sentiment_scores[word.lower()] = sentiment_scores.get(word.lower(), 0) + score
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    return sentiment_scores

# Select initial squad intelligently using price, points, form
def select_initial_squad(data, sentiment):
    players = pd.DataFrame(data['elements'])
    players = players[(players['minutes'] > 0) & (players['now_cost'] > 40)]
    players['score'] = (
        players['total_points'] +
        players['form'].astype(float) * 10 +
        players['bonus'] +
        players['minutes'] / 90
    )
    players = players.sort_values("score", ascending=False)
    selected = players.head(15)["id"].tolist()
    return selected

# Optimize weekly transfers
def optimize_transfers(data, sentiment):
    players = pd.DataFrame(data['elements'])
    players = players[(players['minutes'] > 0) & (players['now_cost'] > 40)]
    players['score'] = (
        players['form'].astype(float) * 10 +
        players['bonus'] +
        players['bps'] +
        players['total_points']
    )
    players = players.sort_values("score", ascending=False)
    top_15 = players.head(15)
    transfer_ids = top_15['id'].tolist()
    captain = top_15.iloc[0]['id']
    vice = top_15.iloc[1]['id']
    return {
        "transfers": [],
        "squad": transfer_ids,
        "captain": captain,
        "vice_captain": vice,
        "chip": None
    }

# Apply transfers to FPL using session
def apply_changes(session, changes, initial=False):
    headers = {"Referer": "https://fantasy.premierleague.com/"}
    payload = {
        "confirmed": True,
        "entry": TEAM_ID,
        "event": get_current_gw_and_deadline()[0],
        "transfers": [],
        "captain": changes['captain'],
        "vice_captain": changes['vice_captain'],
        "chip": changes['chip'] if changes['chip'] else ""
    }
    if initial:
        payload["squad"] = changes["squad"]
    res = session.post(TRANSFERS_URL, headers=headers, json=payload)
    print("Transfer response:", res.status_code)

# Authenticate to FPL
def login_fpl():
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    login_data = {
        'login': EMAIL,
        'password': PASSWORD,
        'redirect_uri': 'https://fantasy.premierleague.com/',
        'app': 'plfpl-web'
    }
    session.post(LOGIN_URL, data=login_data)
    return session

# Export dashboard
def update_dashboard(team):
    df = pd.DataFrame(team)
    df.to_excel("fpl_dashboard.xlsx", index=False)

# Main bot logic
print("\u2699\ufe0f Running ML model...")
def weekly_routine():
    today = datetime.date.today()
    current_gw, deadline = get_current_gw_and_deadline()
    print(f"\n--- GW {current_gw} --- Deadline: {deadline} ---")

    session = login_fpl()
    fpl_data = fetch_fpl_data()
    sentiment = analyze_sentiment()

    # For GW1 with unlimited transfers
    if current_gw == 1 and datetime.date(2025, 8, 10) <= today <= datetime.date(2025, 8, 15):
        print("Creating initial squad for GW1 (unlimited transfers)...")
        initial_squad = select_initial_squad(fpl_data, sentiment)
        captain = initial_squad[0]
        vice_captain = initial_squad[1]
        apply_changes(session, {
            "squad": initial_squad,
            "captain": captain,
            "vice_captain": vice_captain,
            "chip": None
        }, initial=True)
        update_dashboard(initial_squad)
        return

    # After GW1 logic
    print("Optimizing team for upcoming GW...")
    decisions = optimize_transfers(fpl_data, sentiment)
    apply_changes(session, decisions)
    update_dashboard(decisions)

# Schedule bot daily
schedule.every().day.at("07:30").do(weekly_routine)

if __name__ == "__main__":
    print("FPL Auto Bot is running...\n")
    weekly_routine()  # Run once
    while True:
        schedule.run_pending()
        time.sleep(60)
