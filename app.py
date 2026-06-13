import os
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

# =========================================================
# Project settings
# =========================================================
CITY = "Phnom Penh"
CITY_SLUG = CITY.lower().replace(" ", "_")
PREFIX = f"pm25_{CITY_SLUG}"

LATITUDE = 11.5564
LONGITUDE = 104.9282
TIMEZONE = "Asia/Phnom_Penh"
H = 48  # prediction horizon in hours

MODEL_CANDIDATES = [
    f"{PREFIX}_best_model.pkl",        # your notebook output
    f"{PREFIX}_xgboost_model.pkl",     # fallback old name
]
FEATURE_COLS_PATH = f"{PREFIX}_feature_cols.pkl"
FEATURE_SCALER_PATH = f"{PREFIX}_feature_scaler.pkl"  # saved by notebook, not applied for XGBoost/RF
CONFIG_PATH = f"{PREFIX}_config.pkl"                  # optional

# =========================================================
# Page setup + styling
# =========================================================
st.set_page_config(
    page_title="PM2.5 Forecasting — Phnom Penh",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    :root {
        --bg-soft: #f6f8fb;
        --text-main: #172033;
        --text-muted: #667085;
        --border: #e6eaf0;
        --card: #ffffff;
        --primary: #2454ff;
        --primary-soft: #eaf0ff;
        --good: #12b76a;
        --moderate: #f79009;
        --sensitive: #f97316;
        --unhealthy: #ef4444;
        --very: #9f1239;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1220px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f7f9ff 0%, #ffffff 100%);
        border-right: 1px solid var(--border);
    }

    .hero {
        background: radial-gradient(circle at top left, rgba(36, 84, 255, 0.24), transparent 32%),
                    linear-gradient(135deg, #101828 0%, #1d2939 44%, #2447bf 100%);
        color: white;
        padding: 2.1rem 2rem;
        border-radius: 26px;
        margin-bottom: 1.2rem;
        box-shadow: 0 18px 50px rgba(16, 24, 40, 0.18);
    }

    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.22);
        padding: 0.45rem 0.75rem;
        border-radius: 999px;
        font-size: 0.86rem;
        margin-bottom: 1rem;
    }

    .hero h1 {
        font-size: clamp(2.1rem, 4vw, 4rem);
        line-height: 1.02;
        letter-spacing: -0.055em;
        margin: 0 0 0.8rem 0;
    }

    .hero p {
        color: rgba(255, 255, 255, 0.82);
        font-size: 1.05rem;
        max-width: 760px;
        margin: 0;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 760;
        color: #101828;
        margin: 1.2rem 0 0.7rem 0;
    }

    .metric-card, .info-card, .method-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1.15rem 1.2rem;
        box-shadow: 0 8px 24px rgba(16, 24, 40, 0.055);
        height: 100%;
    }

    .metric-label {
        color: var(--text-muted);
        font-size: 0.86rem;
        font-weight: 650;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        color: var(--text-main);
        font-size: 2rem;
        font-weight: 820;
        letter-spacing: -0.035em;
        margin-bottom: 0.15rem;
    }

    .metric-note {
        color: var(--text-muted);
        font-size: 0.84rem;
        line-height: 1.45;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.45rem 0.75rem;
        border-radius: 999px;
        font-weight: 760;
        font-size: 0.92rem;
        margin-top: 0.15rem;
    }

    .badge.good { background: rgba(18, 183, 106, 0.12); color: var(--good); }
    .badge.moderate { background: rgba(247, 144, 9, 0.14); color: var(--moderate); }
    .badge.sensitive { background: rgba(249, 115, 22, 0.14); color: var(--sensitive); }
    .badge.unhealthy { background: rgba(239, 68, 68, 0.13); color: var(--unhealthy); }
    .badge.very { background: rgba(159, 18, 57, 0.13); color: var(--very); }

    .insight-card {
        border-radius: 24px;
        padding: 1.25rem 1.35rem;
        border: 1px solid var(--border);
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        box-shadow: 0 8px 24px rgba(16, 24, 40, 0.055);
        margin-top: 1rem;
    }

    .insight-title {
        font-size: 1.25rem;
        font-weight: 820;
        color: #101828;
        margin-bottom: 0.35rem;
    }

    .insight-body {
        color: #475467;
        font-size: 0.98rem;
        line-height: 1.65;
        margin: 0;
    }

    .tiny-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: #f2f4f7;
        color: #344054;
        border-radius: 999px;
        padding: 0.36rem 0.65rem;
        font-size: 0.82rem;
        font-weight: 650;
        margin: 0.15rem 0.25rem 0.15rem 0;
    }

    .step {
        display: flex;
        gap: 0.8rem;
        align-items: flex-start;
        padding: 0.72rem 0;
        border-bottom: 1px solid #edf0f5;
    }

    .step:last-child { border-bottom: 0; }

    .step-num {
        background: var(--primary-soft);
        color: var(--primary);
        min-width: 1.75rem;
        height: 1.75rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        font-weight: 800;
        font-size: 0.82rem;
    }

    .step-text {
        color: #475467;
        line-height: 1.55;
        font-size: 0.94rem;
    }

    .footer-note {
        color: #667085;
        font-size: 0.86rem;
        line-height: 1.55;
        margin-top: 1rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.45rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 0.55rem 0.95rem;
        background: #f2f4f7;
    }

    .stTabs [aria-selected="true"] {
        background: #eaf0ff !important;
        color: #2447bf !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================================================
# Model loading
# =========================================================
@st.cache_resource
def load_model_assets():
    """Load trained model and feature column order from local files."""
    model_path = None
    for candidate in MODEL_CANDIDATES:
        if os.path.exists(candidate):
            model_path = candidate
            break

    if model_path is None:
        raise FileNotFoundError(
            f"Model file not found. Put `{PREFIX}_best_model.pkl` in the same folder as app.py."
        )

    if not os.path.exists(FEATURE_COLS_PATH):
        raise FileNotFoundError(
            f"Feature column file not found. Put `{FEATURE_COLS_PATH}` in the same folder as app.py."
        )

    model = joblib.load(model_path)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    scaler = joblib.load(FEATURE_SCALER_PATH) if os.path.exists(FEATURE_SCALER_PATH) else None
    config = joblib.load(CONFIG_PATH) if os.path.exists(CONFIG_PATH) else {}

    return model, feature_cols, scaler, config, model_path

# =========================================================
# Open-Meteo data
# =========================================================
def call_api(url: str, params: dict) -> pd.DataFrame:
    """Call Open-Meteo API and return hourly data as a DataFrame."""
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "hourly" not in data or "time" not in data["hourly"]:
        raise ValueError("Open-Meteo response does not contain hourly data.")

    df = pd.DataFrame(data["hourly"])
    df["timestamp"] = pd.to_datetime(df["time"])
    return df.drop(columns=["time"])


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_open_meteo_data(latitude: float, longitude: float) -> pd.DataFrame:
    """
    Fetch recent and forecast hourly air-quality + weather data.
    Forecast rows are included so the 48-hour target timestamp exists.
    Feature engineering still uses only current and past data through positive shifts.
    """
    air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    weather_url = "https://api.open-meteo.com/v1/forecast"

    air_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "pm2_5,pm10",
        "timezone": TIMEZONE,
        "past_hours": 240,
        "forecast_hours": 72,
    }

    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": (
            "temperature_2m,relative_humidity_2m,precipitation,"
            "wind_speed_10m,wind_direction_10m,surface_pressure,cloud_cover"
        ),
        "timezone": TIMEZONE,
        "past_hours": 240,
        "forecast_hours": 72,
    }

    air = call_api(air_url, air_params)
    weather = call_api(weather_url, weather_params)

    df = pd.merge(air, weather, on="timestamp", how="outer").sort_values("timestamp")

    rename_map = {
        "temperature_2m": "temp",
        "relative_humidity_2m": "humidity",
        "precipitation": "rain",
        "wind_speed_10m": "wind_speed",
        "wind_direction_10m": "wind_dir",
        "surface_pressure": "pressure",
        "cloud_cover": "cloud",
    }
    df = df.rename(columns=rename_map)

    needed = [
        "pm2_5", "pm10", "temp", "humidity", "rain",
        "wind_speed", "wind_dir", "pressure", "cloud"
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan

    df[needed] = df[needed].interpolate(limit_direction="both").ffill().bfill()
    return df.reset_index(drop=True)

# =========================================================
# Feature engineering: same methodology as notebook
# =========================================================
def add_methodology_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Create the same no-leakage features from the notebook."""
    feat = df.copy().sort_values("timestamp").reset_index(drop=True)

    cap_pm25 = config.get("cap_pm25") or config.get("PM2.5_cap")
    cap_pm10 = config.get("cap_pm10") or config.get("PM10_cap")
    if cap_pm25 is not None:
        feat["pm2_5"] = feat["pm2_5"].clip(upper=float(cap_pm25))
    if cap_pm10 is not None:
        feat["pm10"] = feat["pm10"].clip(upper=float(cap_pm10))

    # Time features based on target timestamp: the time we are predicting for
    feat["hour"] = feat["timestamp"].dt.hour
    feat["month"] = feat["timestamp"].dt.month
    feat["dayofweek"] = feat["timestamp"].dt.dayofweek

    feat["hour_sin"] = np.sin(2 * np.pi * feat["hour"] / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * feat["hour"] / 24)
    feat["month_sin"] = np.sin(2 * np.pi * (feat["month"] - 1) / 12)
    feat["month_cos"] = np.cos(2 * np.pi * (feat["month"] - 1) / 12)
    feat["dow_sin"] = np.sin(2 * np.pi * feat["dayofweek"] / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * feat["dayofweek"] / 7)
    feat["is_weekend"] = feat["dayofweek"].isin([5, 6]).astype(int)
    feat["is_rush_hour"] = feat["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    cambodian_holidays = [
        (1, 1), (1, 7), (3, 8),
        (4, 14), (4, 15), (4, 16),
        (5, 1), (5, 14), (5, 15), (5, 16),
        (6, 18), (9, 24), (10, 15), (10, 29), (11, 9),
    ]
    month_day = list(zip(feat["timestamp"].dt.month, feat["timestamp"].dt.day))
    feat["is_holiday"] = pd.Series(month_day).isin(cambodian_holidays).astype(int).values

    # PM2.5 lag features: for the target timestamp, shift(H) equals the latest available PM2.5
    for lag in [0, 1, 3, 6, 12, 24, 48, 72, 168]:
        feat[f"pm25_lag_{lag}h"] = feat["pm2_5"].shift(H + lag)

    # Rolling mean/std from prediction time backward
    for window in [3, 6, 12, 24, 48]:
        shifted = feat["pm2_5"].shift(H + 1)
        feat[f"pm25_roll_mean_{window}h"] = shifted.rolling(window).mean()
        feat[f"pm25_roll_std_{window}h"] = shifted.rolling(window).std()

    for lag in [0, 1, 3, 6, 12, 24, 48]:
        feat[f"pm10_lag_{lag}h"] = feat["pm10"].shift(H + lag)

    meteo_cols = ["temp", "humidity", "rain", "wind_speed", "wind_dir", "pressure", "cloud"]
    for col in meteo_cols:
        feat[f"{col}_now"] = feat[col].shift(H)
        feat[f"{col}_lag_12h"] = feat[col].shift(H + 12)
        feat[f"{col}_lag_24h"] = feat[col].shift(H + 24)

    feat["humid_x_wind"] = feat["humidity_now"] * feat["wind_speed_now"]
    feat["temp_x_humidity"] = feat["temp_now"] * feat["humidity_now"]

    return feat

# =========================================================
# UI helpers
# =========================================================
def pm25_category(value: float):
    """Return label, css class, icon, and human-friendly recommendation."""
    if value <= 12:
        return {
            "label": "Good",
            "class": "good",
            "icon": "🟢",
            "message": "Air quality is expected to be comfortable. It should be fine for normal outdoor activities.",
        }
    if value <= 35:
        return {
            "label": "Moderate",
            "class": "moderate",
            "icon": "🟡",
            "message": "Air quality is acceptable, but very sensitive people may want to reduce long outdoor exposure.",
        }
    if value <= 55:
        return {
            "label": "Unhealthy for Sensitive Groups",
            "class": "sensitive",
            "icon": "🟠",
            "message": "People with asthma, heart/lung issues, children, and older adults should be more careful outside.",
        }
    if value <= 150:
        return {
            "label": "Unhealthy",
            "class": "unhealthy",
            "icon": "🔴",
            "message": "Consider limiting outdoor exercise, closing windows, and using a mask or air purifier if available.",
        }
    return {
        "label": "Very Unhealthy",
        "class": "very",
        "icon": "🟣",
        "message": "Avoid long outdoor activities. Sensitive groups should stay indoors as much as possible.",
    }


def metric_card(label: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_card(label: str, level_class: str, icon: str, note: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Air quality category</div>
            <div class="badge {level_class}">{icon} {label}</div>
            <div class="metric-note" style="margin-top:0.75rem;">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def setup_error_panel(error_message: str):
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">⚙️ Setup required</div>
            <h1>Almost ready.</h1>
            <p>The app design is ready, but the trained model files must be placed beside <b>app.py</b> before prediction can run.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.error(error_message)
    st.markdown("### Put these files in the same folder as `app.py`")
    st.code(
        f"""{PREFIX}_best_model.pkl
{PREFIX}_feature_cols.pkl
{PREFIX}_feature_scaler.pkl""",
        language="text",
    )
    st.markdown("Then run:")
    st.code("streamlit run app.py", language="bash")


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.markdown("### 🌫️ Forecast settings")
    st.caption("This app uses your trained model and reads data automatically from Open-Meteo.")

    latitude = st.number_input("Latitude", value=LATITUDE, format="%.4f")
    longitude = st.number_input("Longitude", value=LONGITUDE, format="%.4f")

    st.markdown("---")
    st.markdown("#### Model setup")
    st.write(f"**City:** {CITY}")
    st.write(f"**Prediction horizon:** {H} hours ahead")
    st.write(f"**Timezone:** {TIMEZONE}")

    refresh = st.button("🔄 Refresh live data", use_container_width=True)
    if refresh:
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("#### How it works")
    st.caption(
        "The app fetches recent PM2.5, PM10, and weather data, "
    )

# =========================================================
# App body
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">🌏 Phnom Penh · Live data · Machine learning forecast</div>
        <h1>PM2.5 air quality forecast</h1>
        <p>
            This app provides a live forecast of PM2.5 air quality in Phnom Penh based on recent weather and pollution data.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, feature_cols, scaler, config, model_path = load_model_assets()
    model_class = type(model).__name__

    with st.spinner("Fetching Open-Meteo data and preparing model features..."):
        raw_df = fetch_open_meteo_data(latitude, longitude)
        feature_df = add_methodology_features(raw_df, config)

    local_now = pd.Timestamp.now(tz=ZoneInfo(TIMEZONE)).tz_localize(None).floor("h")
    prediction_time = raw_df.loc[raw_df["timestamp"] <= local_now, "timestamp"].max()
    target_time = prediction_time + pd.Timedelta(hours=H)

    target_rows = feature_df[feature_df["timestamp"] == target_time]
    if target_rows.empty:
        raise ValueError("Target timestamp is not available from Open-Meteo yet. Try refreshing later.")

    row = target_rows.iloc[0]
    X = pd.DataFrame([row]).reindex(columns=feature_cols).astype(float)

    if X.isna().any().any():
        missing = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Some required model features are missing: {missing[:10]}")

    # Important: your XGBoost/Random Forest model was trained on raw X_train values, not scaled values.
    prediction = float(model.predict(X.values)[0])
    current_pm25 = float(raw_df.loc[raw_df["timestamp"] == prediction_time, "pm2_5"].iloc[0])
    current_pm10 = float(raw_df.loc[raw_df["timestamp"] == prediction_time, "pm10"].iloc[0])
    level = pm25_category(prediction)

    st.markdown('<div class="section-title">Prediction summary</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([1.05, 1.2, 1.05, 1.05])
    with col1:
        metric_card("Predicted PM2.5", f"{prediction:.1f}", "µg/m³ · 48 hours ahead")
    with col2:
        status_card(level["label"], level["class"], level["icon"], level["message"])
    with col3:
        metric_card("Target time", target_time.strftime("%d %b %Y"), target_time.strftime("%I:%M %p"))
    with col4:
        metric_card("Current PM2.5", f"{current_pm25:.1f}", f"PM10 now: {current_pm10:.1f} µg/m³")

    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">{level['icon']} What this prediction means</div>
            <p class="insight-body">
                The model expects PM2.5 in <b>{CITY}</b> to be around <b>{prediction:.1f} µg/m³</b>
                at <b>{target_time:%A, %d %B %Y, %I:%M %p}</b>. This is classified as
                <b>{level['label']}</b>. The prediction was generated using data available up to
                <b>{prediction_time:%A, %d %B %Y, %I:%M %p}</b>, so the app does not use future PM2.5 values as input.
            </p>
            <div style="margin-top:0.9rem;">
                <span class="tiny-pill"> Model: {model_class}</span>
                <span class="tiny-pill"> File: {model_path}</span>
                <span class="tiny-pill"> Horizon: {H} hours</span>
                <span class="tiny-pill"> Source: Open-Meteo</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trend", "🌦️ Weather", "🧠 Methodology", "🔍 Model input"])

    with tab1:
        st.markdown('<div class="section-title">Recent pollution trend</div>', unsafe_allow_html=True)
        chart_df = raw_df[raw_df["timestamp"] <= prediction_time].tail(96).set_index("timestamp")
        st.line_chart(chart_df[["pm2_5", "pm10"]], use_container_width=True)
        st.caption("The chart shows recent hourly PM2.5 and PM10 values from Open-Meteo. These recent values are used to build lag and rolling features.")

        recent = chart_df[["pm2_5", "pm10"]].tail(24)
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("24h average PM2.5", f"{recent['pm2_5'].mean():.1f}", "µg/m³")
        with c2:
            metric_card("24h max PM2.5", f"{recent['pm2_5'].max():.1f}", "µg/m³")
        with c3:
            metric_card("24h average PM10", f"{recent['pm10'].mean():.1f}", "µg/m³")

    with tab2:
        st.markdown('<div class="section-title">Current weather conditions used by the model</div>', unsafe_allow_html=True)
        weather_cols = ["temp", "humidity", "rain", "wind_speed", "pressure", "cloud"]
        latest_weather = raw_df[raw_df["timestamp"] == prediction_time].iloc[0]
        w1, w2, w3 = st.columns(3)
        with w1:
            metric_card("Temperature", f"{latest_weather['temp']:.1f}°C", "Open-Meteo hourly weather")
        with w2:
            metric_card("Humidity", f"{latest_weather['humidity']:.0f}%", "Relative humidity")
        with w3:
            metric_card("Wind speed", f"{latest_weather['wind_speed']:.1f}", "km/h")

        weather_chart = raw_df[raw_df["timestamp"] <= prediction_time].tail(72).set_index("timestamp")
        st.line_chart(weather_chart[weather_cols], use_container_width=True)
        st.caption("Weather can affect PM2.5 concentration because wind, humidity, rain, and pressure influence how pollution moves or stays in the air.")

    with tab3:
        st.markdown('<div class="section-title">Methodology from your project</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="method-card">
                <div class="step"><span class="step-num">1</span><div class="step-text"><b>Collect data automatically.</b> The app reads hourly PM2.5, PM10, and weather data from Open-Meteo, so users do not upload a dataset.</div></div>
                <div class="step"><span class="step-num">2</span><div class="step-text"><b>Clean and prepare values.</b> Missing hourly values are filled carefully so the lag and rolling features can be calculated.</div></div>
                <div class="step"><span class="step-num">3</span><div class="step-text"><b>Create time features.</b> The app adds hour, month, day of week, weekend, rush hour, and Cambodian holiday indicators.</div></div>
                <div class="step"><span class="step-num">4</span><div class="step-text"><b>Create no-leakage lag features.</b> PM2.5, PM10, and weather variables are shifted so the model only uses current and past information.</div></div>
                <div class="step"><span class="step-num">5</span><div class="step-text"><b>Predict 48 hours ahead.</b> The saved best model predicts the PM2.5 level for the target time.</div></div>
                <div class="step"><span class="step-num">6</span><div class="step-text"><b>Explain the result.</b> The app shows the predicted value, air-quality category, trend chart, and simple recommendation.</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if scaler is not None:
            st.info(
                f"`{FEATURE_SCALER_PATH}` was found, but it is not applied here because your XGBoost/Random Forest training used raw feature values."
            )

    with tab4:
        st.markdown('<div class="section-title">Feature row sent to the model</div>', unsafe_allow_html=True)
        st.caption("This table is useful for debugging and presentation. It shows the exact feature order used for prediction.")
        st.dataframe(X.T.rename(columns={0: "value"}), use_container_width=True, height=520)

    st.markdown(
        """
        <p class="footer-note">
            Note: This project is for educational and forecasting purposes. Air-quality predictions may change when new weather and pollution data become available.
        </p>
        """,
        unsafe_allow_html=True,
    )

except Exception as e:
    setup_error_panel(str(e))
