# 🧼 Wash Trading Detector using Bitquery + XGBoost

This project detects potential wash trading behavior on DEX platforms using on-chain Solana trade data from Bitquery. It uses a combination of rule-based heuristics and an XGBoost machine learning model to flag suspicious trades.

## Built with:

- Bitquery GraphQL API (Solana DEXTrades)
- XGBoost for ML classification
- Streamlit for an interactive demo app

## 🚀 Features

- Fetches real-time Solana DEX trade data from Bitquery
- Applies rules to identify suspicious behavior (self-trades, spoofing, trade loops)
- Trains an XGBoost model on labeled trade data
- Deployable via Streamlit Cloud
- Supports JSON file upload for custom trade analysis

## 🛠️ Setup Instructions

1. Clone the repository
git clone https://github.com/Kshitij0O7/wash-trading-detector
cd wash-trading-detector
2. Install dependencies
pip install -r requirements.txt
3. Add Bitquery API key
For local testing, create a file at .streamlit/secrets.toml:

BITQUERY_API_KEY = "your-bitquery-api-key"
For Streamlit Cloud, add this in your Secrets Manager.

▶️ Running the App Locally

streamlit run app.py
🧠 Project Structure

app.py                # Streamlit web app
get_data.py           # Bitquery API integration
label.py              # Rule-based trade labeling functions
model_features.json   # Feature order used for prediction
xgb_wash_model.pkl    # Trained XGBoost model
requirements.txt      # Python dependencies
.streamlit/
  └── secrets.toml    # (optional, local secret config)
🌐 Deploy on Streamlit Cloud

Push this repo to GitHub
Connect to Streamlit Cloud
Set your BITQUERY_API_KEY in the Secrets tab
Done!
📄 License

MIT License

Let me know if you want a version that includes screenshots or badges (e.g., "Deploy to Streamlit" or "Made with Bitquery").