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
print("📦 Downloading NLTK resources...")
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
from openpyxl import Workbook
print("✅ Bot started execution.")

# Load environment variables
load_dotenv()
EMAIL = os.getenv("FPL_EMAIL")
PASSWORD = os.getenv("FPL_PASSWORD")
TEAM_ID = os.getenv("FPL_TEAM_ID")
print("🔐 Logging into FPL...")
# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# --- PLACEHOLDER FUNCTIONS ---
# In the real deployment, implement these modules properly.

def fetch_fpl_data():
    # Pull official FPL data
    return requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
print("📊 Fetching player stats...")
def get_current_gw_and_deadline():
    events = fetch_fpl_data()["events"]
    for event in events:
        if event["is_current"]:
            deadline = datetime.datetime.strptime(event["deadline_time"], "%Y-%m-%dT%H:%M:%SZ")
            return event["id"], deadline
    return None, None

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
print("📰 Scraping media news...")
def select_initial_squad(data, sentiment):
    print("Selecting initial squad...")
    return [101, 102, 103]  # Dummy player IDs

def optimize_transfers(data, sentiment):
    return {
        "transfers": [(201, 202)],  # out_id, in_id
        "captain": 301,
        "vice_captain": 302,
        "chip": None
    }

def update_dashboard(team):
    df = pd.DataFrame(team)
    df.to_excel("fpl_dashboard.xlsx", index=False)

def apply_changes(changes):
    print("Applying transfers:", changes)
    # Integrate with login and transfer logic (see fantasy.premierleague API)

def initial_team_exists():
    # Stub logic; replace with real check later
    return False
print("⚙️ Running ML model...")
# --- CORE LOOP ---
def weekly_routine():
    today = datetime.date.today()
    current_gw, deadline = get_current_gw_and_deadline()
    print(f"Current GW: {current_gw}, Deadline: {deadline}")

    fpl_data = fetch_fpl_data()
    sentiment = analyze_sentiment()

    if current_gw == 1 and not initial_team_exists():
        if datetime.date(2025, 8, 10) <= today <= datetime.date(2025, 8, 15):
            initial_squad = select_initial_squad(fpl_data, sentiment)
            apply_changes({"initial_squad": initial_squad})
            update_dashboard(initial_squad)
            return
        else:
            print("Waiting for appropriate window to build initial squad.")
            return

    decisions = optimize_transfers(fpl_data, sentiment)
    apply_changes(decisions)
    update_dashboard(decisions)

# Schedule the bot daily to catch all deadline variations
schedule.every().day.at("07:30").do(weekly_routine)
print("🧠 Making transfers...")
if __name__ == "__main__":
    print("FPL Auto Bot is running...")
    weekly_routine()  # run once immediately
    while True:
        schedule.run_pending()
        time.sleep(60)
