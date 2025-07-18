import pandas as pd
from token_rules import (
    detect_self_trades,
    detect_repeated_pairs,
    detect_loops,
    detect_spoofing,
    get_suspicious_summary
)

def label_trades(trade_data):
    df = pd.json_normalize(trade_data)

    self_trades = detect_self_trades(df)
    repeated_pairs = detect_repeated_pairs(df)
    loops = detect_loops(df)
    spoofing = detect_spoofing(df)

    suspicious_tokens, suspicious_tx, suspicious_wallets = get_suspicious_summary(
        self_df=self_trades,
        repeated_df=repeated_pairs,
        loops_df=loops,
        spoofed_df=spoofing,
        original_df=df
    )

    df["is_wash_trade"] = (
        df["Trade.Buy.Account.Address"].isin(suspicious_wallets) |
        df["Trade.Sell.Account.Address"].isin(suspicious_wallets) |
        df["Transaction.Signature"].isin(suspicious_tx)
    )

    return df
