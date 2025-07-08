
from flask import Flask, render_template, request, url_for
import joblib
import pandas as pd

app = Flask(__name__)

player_model = joblib.load("models/points_predictor.pkl")
player_data = pd.read_csv("data/combined_player_game_logs.csv")

player_image_ids = {
    "lebron james": "2544",
    "stephen curry": "201939",
    "kevin durant": "201142",
    "giannis antetokounmpo": "203507"
}

def get_player_image_url(name):
    name_lower = name.lower()
    player_id = player_image_ids.get(name_lower)
    if player_id:
        return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
    return url_for('static', filename='default_player.png')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    player_name = request.form["player_name"]
    opponent = request.form["opponent"]
    player_img_url = get_player_image_url(player_name)

    recent = player_data[player_data["Player"].str.lower() == player_name.lower()]
    if recent.empty:
        return render_template("index.html", result="No data available for this player.", player_img_url=player_img_url)

    recent_game = recent.iloc[-1]
    input_data = pd.DataFrame([{
        'Minutes': recent_game['Minutes'],
        'Assists': recent_game['Assists'],
        'Rebounds': recent_game['Rebounds'],
        'Home': 1 if recent_game['Home/Away'] == 'Home' else 0
    }])
    prediction = player_model.predict(input_data)[0]
    result = f"{player_name} is predicted to score {prediction:.1f} points vs {opponent}"
    return render_template("index.html", result=result, player_img_url=player_img_url)
