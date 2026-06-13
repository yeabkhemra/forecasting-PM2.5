# PM2.5 Forecasting Streamlit App

This app deploys your PM2.5 Phnom Penh 48-hour forecasting model.
It does not require dataset upload. It fetches hourly PM2.5, PM10, and weather data from Open-Meteo.

## Required files from your notebook

Your notebook saving code creates these files:

```text
pm25_phnom_penh_best_model.pkl
pm25_phnom_penh_feature_cols.pkl
pm25_phnom_penh_feature_scaler.pkl
```

Put those files in the same folder as `app.py`.

Important: for XGBoost and Random Forest, the app does **not** apply `feature_scaler.pkl`, because your notebook trained those models using raw `X_train` values. The scaler file is only loaded for compatibility.

## Run locally

```bash
cd pm25_streamlit_xgboost_app
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Upload these files to GitHub:

```text
app.py
requirements.txt
pm25_phnom_penh_best_model.pkl
pm25_phnom_penh_feature_cols.pkl
pm25_phnom_penh_feature_scaler.pkl
```

Then choose `app.py` as the main file in Streamlit Cloud.
