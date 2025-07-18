def detect_self_trades(df):
    return df[df["Trade.Buy.Account.Address"] == df["Trade.Sell.Account.Address"]]

def detect_repeated_pairs(df, threshold=5):
    pairs = df.groupby([
        "Trade.Buy.Account.Address",
        "Trade.Sell.Account.Address"
    ]).size().reset_index(name="count")
    return pairs[pairs["count"] > threshold]

def detect_loops(df):
    merged = df.merge(df, left_on="Trade.Buy.Account.Address", right_on="Trade.Sell.Account.Address")
    loops = merged[merged["Trade.Sell.Account.Address_x"] == merged["Trade.Buy.Account.Address_y"]]
    return loops

def detect_spoofing(df, price_threshold=2.0):
    df["spread"] = abs(df["Trade.Buy.PriceInUSD"] - df["Trade.Sell.PriceInUSD"])
    return df[df["spread"] > price_threshold]

def get_suspicious_summary(self_df, repeated_df, loops_df, spoofed_df, original_df):
    wallets = set(self_df["Trade.Buy.Account.Address"])
    wallets |= set(repeated_df["Trade.Buy.Account.Address"])
    wallets |= set(loops_df["Trade.Buy.Account.Address_x"])
    wallets |= set(spoofed_df["Trade.Buy.Account.Address"])

    suspicious_trades = original_df[
        original_df["Trade.Buy.Account.Address"].isin(wallets) |
        original_df["Trade.Sell.Account.Address"].isin(wallets)
    ]

    suspicious_tokens = suspicious_trades["Trade.Buy.Currency.MintAddress"].unique().tolist()
    suspicious_tx = suspicious_trades["Transaction.Signature"].unique().tolist()

    return suspicious_tokens, suspicious_tx, wallets
