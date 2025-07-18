import requests
import json
import streamlit as st

def get_trades():
    token = st.secrets["token"]
    url = "https://streaming.bitquery.io/eap"

    query = """
    {
        Solana {
            DEXTrades(
                orderBy: {descending: Block_Time}
            ) {
                Trade {
                    Dex {
                        ProtocolName
                        ProtocolFamily
                    }
                    Buy {
                        Account{
                            Address
                        }
                        Amount
                        AmountInUSD
                        Currency {
                            Symbol
                            Name
                            MintAddress
                        }
                        PriceInUSD
                    }
                    Sell {
                        Account{
                            Address
                        }
                        Amount
                        AmountInUSD
                        Currency {
                            Symbol
                            Name
                            MintAddress
                        }
                        PriceInUSD
                    }
                }
                Block {
                    Time
                    Height
                }
                Transaction {
                    Signature
                    FeePayer
                }
            }
        }
    }
    """

    payload = json.dumps({
        "query": query,
        "variables": "{}"
    })

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer ${token}'
    }

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        data = response.json()

        # Navigate the JSON structure to get the DEX trade list
        trades = data["data"]["Solana"]["DEXTrades"]
        # print(trades)
        return trades

    except (requests.exceptions.RequestException, KeyError, json.JSONDecodeError) as e:
        print(f"Error fetching DEX trades: {e}")
        return []