import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
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
wash_count = int(df["prediction"].sum())
total_trades = len(df)
risk_score = min(100, int((wash_count / total_trades) * 100)) if total_trades else 0

# ---- Streamlit UI ----
st.set_page_config(page_title="Solana Wash Trade Risk Dashboard", layout="wide")

st.title("🚨 Solana Wash Trading Risk Assessment")

st.button("🔍 Analyze Wash Trading Risk for Solana")

# 🔵 Risk Score Section
st.subheader("Risk Assessment")
st.markdown(f"**Risk Score: `{risk_score}`** – {'High' if risk_score > 70 else 'Medium' if risk_score > 40 else 'Low'} Risk")
st.progress(risk_score / 100)

# Feilds for display and calculations
df["timestamp"] = pd.to_datetime(df["Block.Time"], errors="coerce")
df["volume"] = pd.to_numeric(df["Trade.Buy.AmountInUSD"], errors="coerce").fillna(0)
df["suspiciousVolume"] = df.apply(
    lambda row: row["volume"] if row["prediction"] == 1 else 0,
    axis=1
)
# Group by buyer and sum their volumes
top_wallets = df.groupby("Trade.Buy.Account.Address")["volume"].sum().sort_values(ascending=False)

# Compute total and top 10 share
total_volume = top_wallets.sum()
top_10_volume = top_wallets.head(10).sum()
volume_concentration = round(100 * top_10_volume / total_volume, 2)

top_wallet_trade_count = df["Trade.Buy.Account.Address"].value_counts().iloc[0]
wallet_contribution = round(100 * top_wallet_trade_count / len(df), 2)

wash_volume_pct = round(100 * df["suspiciousVolume"].sum() / df["volume"].sum(), 2)

# 🔵 Stat Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Volume Concentration", f"{volume_concentration}%")
col2.metric("Most Active Trader", f"{wallet_contribution}%")
col3.metric("Suspicious Trades", f"{wash_count}")
col4.metric("Wash Trade Volume", f"{wash_volume_pct}%")

df.set_index("timestamp", inplace=True)
agg = df.resample("1s").agg({
    "volume": "sum",
    "suspiciousVolume": "sum"
}).fillna(0).reset_index()

fig = go.Figure()

# Blue bars = total volume
fig.add_trace(go.Bar(
    x=agg["timestamp"],
    y=agg["volume"],
    name="Total Volume",
    marker_color="blue",
    width=500  # ~60 seconds in ms to avoid overlap
))

# Red bars = suspicious volume
fig.add_trace(go.Bar(
    x=agg["timestamp"],
    y=agg["suspiciousVolume"],
    name="Suspicious Volume",
    marker_color="red",
    width=500
))

fig.update_layout(
    barmode="group",
    title="Grouped Bar Chart: Volume vs Suspicious Volume",
    xaxis_title="Time(UTC+00:00:00)",
    yaxis_title="Volume (USD)",
    xaxis_tickformat="%H:%M:%S",
    legend=dict(x=0.8, y=1.1),
    bargap=0.4,
    bargroupgap=0.1,
    height=400,
)

st.plotly_chart(fig, use_container_width=True)

# print(df.index.min(), df.index.max())
# print(f"Span: {(df.index.max() - df.index.min()).total_seconds()} seconds")

# 🔵 Show full raw data
if st.checkbox("Show Raw Trade Data"):
    st.dataframe(df)