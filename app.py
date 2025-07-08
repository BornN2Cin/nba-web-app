
from flask import Flask, render_template, request, url_for
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

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

def train_model():
    X = [
        [30, 5, 8, 1],
        [35, 7, 10, 0],
        [32, 6, 4, 1]
    ]
    y = [25, 30, 28]
    model = LinearRegression().fit(X, y)
    return model

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
    input_data = [[
        recent_game["Minutes"],
        recent_game["Assists"],
        recent_game["Rebounds"],
        1 if recent_game["Home/Away"] == "Home" else 0
    ]]
    model = train_model()
    prediction = model.predict(input_data)[0]
    result = f"{player_name} is predicted to score {prediction:.1f} points vs {opponent}"
    return render_template("index.html", result=result, player_img_url=player_img_url)

@app.route('/rookie')
def rookie():
    return render_template("rookie.html")

@app.route('/charts')
def charts():
    return render_template("charts.html")
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
