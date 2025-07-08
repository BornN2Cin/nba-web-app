
from flask import Flask, render_template, request, url_for
import os
import pandas as pd
import joblib
import sqlite3
from datetime import datetime
from sklearn.linear_model import LinearRegression

app = Flask(__name__, instance_relative_config=True)

# Load model (replace with your real model if needed)
def train_model():
    X = [
        [30, 5, 8, 1],
        [35, 7, 10, 0],
        [32, 6, 4, 1]
    ]
    y = [25, 30, 28]
    return LinearRegression().fit(X, y)

player_model = train_model()

# Load sample player data
try:
    player_data = pd.read_csv("data/combined_player_game_logs.csv")
except:
    player_data = pd.DataFrame([{
        "Player": "LeBron James",
        "Minutes": 35,
        "Assists": 8,
        "Rebounds": 7,
        "Home/Away": "Home",
        "Game Date": "2025-04-12"
    }])

player_image_ids = {
    "lebron james": "2544",
    "stephen curry": "201939",
    "kevin durant": "201142",
    "giannis antetokounmpo": "203507"
}

team_logo_urls = {
    "Lakers": "/static/logos/lakers.png",
    "Warriors": "/static/logos/warriors.png"
    # Add more teams here
}

nba_teams = [
    "Lakers", "Warriors", "Celtics", "Bucks", "Heat", "Suns", "Nuggets", "Clippers",
    "76ers", "Bulls", "Knicks", "Mavericks", "Raptors", "Hawks", "Grizzlies", "Nets",
    "Cavaliers", "Pelicans", "Kings", "Timberwolves", "Hornets", "Magic", "Wizards",
    "Pistons", "Pacers", "Thunder", "Jazz", "Spurs", "Trail Blazers", "Rockets"
]

def get_player_image_url(name):
    name_lower = name.lower()
    player_id = player_image_ids.get(name_lower)
    if player_id:
        return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
    return url_for('static', filename='default_player.png')

def log_prediction(player_name, opponent, predicted_points, game_date=None):
    db_path = os.path.join(app.instance_path, "nba_predictions.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (player_name, opponent, predicted_points, game_date)
        VALUES (?, ?, ?, ?)
    """, (player_name, opponent, predicted_points, game_date))
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template("index.html", teams=nba_teams, year=datetime.now().year)

@app.route('/predict', methods=["POST"])
def predict():
    player_name = request.form["player_name"]
    opponent = request.form["opponent"]
    player_img_url = get_player_image_url(player_name)
    team_logo_url = team_logo_urls.get(opponent, None)

    recent = player_data[player_data["Player"].str.lower() == player_name.lower()]
    if recent.empty:
        return render_template("index.html", result="No data available for this player.",
                               player_img_url=player_img_url, teams=nba_teams, year=datetime.now().year)

    recent_game = recent.iloc[-1]
    input_data = [[
        recent_game["Minutes"],
        recent_game["Assists"],
        recent_game["Rebounds"],
        1 if recent_game["Home/Away"] == "Home" else 0
    ]]
    prediction = player_model.predict(input_data)[0]

    # Log prediction
    log_prediction(player_name, opponent, round(prediction, 1), recent_game.get("Game Date", None))

    result = f"{player_name} is predicted to score {prediction:.1f} points vs {opponent}"
    return render_template("index.html", result=result, player_img_url=player_img_url,
                           team_logo_url=team_logo_url, teams=nba_teams, year=datetime.now().year)

@app.route('/rookie')
def rookie():
    return render_template("rookie.html")

@app.route('/charts')
def charts():
    return render_template("charts.html")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

