import pandas as pd
from get_data import get_trades
from label import label_trades
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import json

trade_data = get_trades()
df = label_trades(trade_data)

for col in df.columns:
    if df[col].dtype == 'object' and col != 'is_wash_trade':
        df[col] = df[col].astype('category')

X = df.drop(columns=["is_wash_trade"])
y = df["is_wash_trade"]

with open("model_features.json", "w") as f:
    json.dump(X.columns.tolist(), f)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(enable_categorical=True, tree_method='hist')
model.fit(X_train, y_train)
with open("xgb_wash_model.pkl", "wb") as f:
    pickle.dump(model, f)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
