import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import json
from get_data import get_trades

# Load the trained model
model = pickle.load(open("xgb_wash_model.pkl", "rb"))

# Load the training feature column list
with open("model_features.json", "r") as f:
    feature_cols = json.load(f)

# Load fresh data
trade_data = get_trades()
df = pd.json_normalize(trade_data)

# Prepare features
X = df.copy()

# Encode categorical features
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Align with training features
for col in feature_cols:
    if col not in X.columns:
        X[col] = 0
X = X[feature_cols]

# Predict
df["prediction"] = model.predict(X)

# ---- Streamlit UI ----
st.title("🚨 Wash Trade Detector Demo")
st.write("Using on-chain DEX data from Bitquery + XGBoost model")

# Stats
st.metric("Total Trades", len(df))
st.metric("Predicted Wash Trades", int(df["prediction"].sum()))

# Chart
chart_data = df["prediction"].value_counts().rename({0: "Normal", 1: "Wash Trade"})
st.bar_chart(chart_data)

# Show data toggle
if st.checkbox("Show raw data"):
    st.dataframe(df)

# Upload new trade data (optional)
uploaded = st.file_uploader("Upload new trade JSON", type=["json"])
if uploaded:
    new_data = pd.read_json(uploaded)
    new_df = pd.json_normalize(new_data)
    X_new = new_df.copy()

    for col in X_new.select_dtypes(include="object").columns:
        X_new[col] = LabelEncoder().fit_transform(X_new[col].astype(str))

    for col in feature_cols:
        if col not in X_new.columns:
            X_new[col] = 0
    X_new = X_new[feature_cols]

    new_df["prediction"] = model.predict(X_new)
    st.dataframe(new_df)
