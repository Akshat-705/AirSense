from pathlib import Path
import os
from datetime import datetime, timezone

import requests
from flask import Flask, jsonify, render_template, request
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

app = Flask(__name__, template_folder=".")
OPENWEATHER_API_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
OPENWEATHER_GEOCODING_URL = "https://api.openweathermap.org/geo/1.0/direct"

# =========================
# LOAD DATA
# =========================
data = pd.read_csv(Path(__file__).with_name("city_data.csv"))
features = ["PM2.5", "PM10", "NO2", "CO", "SO2"]
target = "AQI"
data[features + [target]] = data[features + [target]].apply(pd.to_numeric, errors="coerce")
data = data.dropna(subset=[target]).copy()

# =========================
# ML MODELS AND EVALUATION
# =========================
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(data[features]), columns=features, index=data.index)
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=150, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Linear Regression": LinearRegression(),
}
evaluation = {}
for name, candidate in models.items():
    candidate.fit(X_train, y_train)
    predictions = candidate.predict(X_test)
    evaluation[name] = {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
        "r2": r2_score(y_test, predictions),
    }

model_name = min(evaluation, key=lambda name: evaluation[name]["mae"])
model = models[model_name]
model.fit(X, y)
model_metrics = evaluation[model_name]
feature_importance = []
if hasattr(model, "feature_importances_"):
    feature_importance = sorted(
        zip(features, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
else:
    feature_importance = sorted(
        zip(features, abs(model.coef_)),
        key=lambda item: item[1],
        reverse=True,
    )


def fetch_live_air_quality(city):
    """Fetch current pollutant readings and estimate AQI for a city."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENWEATHER_API_KEY is not configured. Set it before requesting live data."
        )

    geocoding_response = requests.get(
        OPENWEATHER_GEOCODING_URL,
        params={"q": f"{city},IN", "limit": 1, "appid": api_key},
        timeout=10,
    )
    geocoding_response.raise_for_status()
    locations = geocoding_response.json()
    if not locations:
        raise LookupError(f"OpenWeather could not locate {city}.")

    location = locations[0]
    pollution_response = requests.get(
        OPENWEATHER_API_URL,
        params={"lat": location["lat"], "lon": location["lon"], "appid": api_key},
        timeout=10,
    )
    pollution_response.raise_for_status()
    payload = pollution_response.json()
    if not payload.get("list"):
        raise LookupError(f"OpenWeather returned no air-quality data for {city}.")

    reading = payload["list"][0]
    components = reading.get("components", {})
    live_features = pd.DataFrame(
        [[
            components.get("pm2_5"),
            components.get("pm10"),
            components.get("no2"),
            components.get("co", 0) / 1000,
            components.get("so2"),
        ]],
        columns=features,
    )
    if live_features.isna().all(axis=None):
        raise LookupError(f"OpenWeather returned incomplete air-quality data for {city}.")

    prepared_features = pd.DataFrame(
        imputer.transform(live_features), columns=features
    )
    estimated_aqi = float(model.predict(prepared_features)[0])
    return {
        "city": city,
        "estimated_aqi": round(estimated_aqi, 2),
        "provider_index": reading.get("main", {}).get("aqi"),
        "pm25": round(float(prepared_features.iloc[0]["PM2.5"]), 2),
        "pm10": round(float(prepared_features.iloc[0]["PM10"]), 2),
        "updated_at": datetime.fromtimestamp(
            reading["dt"], tz=timezone.utc
        ).isoformat(),
        "source": "OpenWeather Air Pollution API",
    }


# =========================
# ANALYSIS FUNCTION
# =========================
def generate_analysis(aqi_values):

    avg_aqi = sum(aqi_values) / len(aqi_values)
    max_aqi = max(aqi_values)
    min_aqi = min(aqi_values)

    if avg_aqi <= 50:
        trend = "Overall air quality is GOOD 😊"
    elif avg_aqi <= 100:
        trend = "Air quality is MODERATE 😐"
    else:
        trend = "Air quality is POOR 😷"

    if max_aqi > 400:
        warning = "⚠️ Severe pollution detected on some days!"
    else:
        warning = "No extreme pollution spikes."

    analysis = f"""
    <b>Average AQI:</b> {round(avg_aqi,2)} <br>
    <b>Maximum AQI:</b> {max_aqi} <br>
    <b>Minimum AQI:</b> {min_aqi} <br><br>
    <b>Status:</b> {trend} <br>
    <b>Warning:</b> {warning}
    """

    return analysis

# =========================
# HOME
# =========================
@app.route('/')
def home():
    cities = list(data['City'].unique())
    return render_template('Index.html', cities=cities)

# =========================
# FILTER
# =========================
@app.route('/filter', methods=['POST'])
def filter_data():

    city = request.form.get("city", "")
    filtered = data[data["City"] == city].sort_values("Date")

    if filtered.empty:
        return "No data found for the selected city", 404

    dates = list(filtered["Date"].astype(str))
    aqi = list(filtered["AQI"].round(2))
    pm25 = list(filtered["PM2.5"].fillna(X["PM2.5"].median()).round(2))
    pm10 = list(filtered["PM10"].fillna(X["PM10"].median()).round(2))

    latest_row = filtered.tail(1).iloc[0]

    latest_features = pd.DataFrame(
        [[latest_row[feature] for feature in features]], columns=features
    )
    latest_features = pd.DataFrame(
        imputer.transform(latest_features), columns=features
    )
    latest_aqi = float(latest_row[target])
    pm25_val = float(latest_features.iloc[0]["PM2.5"])
    pm10_val = float(latest_features.iloc[0]["PM10"])

    # ML Prediction
    pred = model.predict(latest_features)[0]
    pred = float(pred)

    # =========================
    # HEALTH ADVICE + PRECAUTIONS
    # =========================
    if pred <= 50:
        advice = "Good 😊 Safe to go outside"
        precautions = [
            "Enjoy outdoor activities",
            "No health risk"
        ]
        color = "green"

    elif pred <= 100:
        advice = "Moderate 😐 Limit outdoor activities"
        precautions = [
            "Reduce prolonged outdoor exposure",
            "Sensitive people should be careful"
        ]
        color = "orange"

    else:
        advice = "Poor 😷 Health Risk!"
        precautions = [
            "Stay indoors",
            "Wear N95 mask if going outside",
            "Avoid exercise",
            "Use air purifier"
        ]
        color = "red"

    # ANALYSIS
    analysis = generate_analysis(aqi)

    return render_template(
        'Index.html',
        cities=list(data['City'].unique()),
        selected_city=city,
        dates=dates,
        aqi=aqi,
        pm25=pm25,
        pm10=pm10,
        latest_aqi=latest_aqi,
        pm25_val=pm25_val,
        pm10_val=pm10_val,
        pred=round(pred, 2),
        advice=advice,
        color=color,
        analysis=analysis,
        precautions=precautions,
        model_name=model_name,
        model_mae=round(model_metrics["mae"], 2),
        model_rmse=round(model_metrics["rmse"], 2),
        model_r2=round(model_metrics["r2"], 3),
        feature_importance=[
            {"name": name, "value": round(value * 100, 1)}
            for name, value in feature_importance
        ],
    )


@app.route("/api/live/<city>")
def live_air_quality(city):
    if city not in set(data["City"].unique()):
        return jsonify({"error": "Unsupported city"}), 404
    try:
        return jsonify(fetch_live_air_quality(city))
    except (requests.RequestException, LookupError, RuntimeError) as error:
        app.logger.warning("Live AQI request failed for %s: %s", city, error)
        return jsonify({"error": str(error)}), 503


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)
