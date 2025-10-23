import os, requests, pandas as pd
from datetime import datetime, timedelta

ALPACA_KEY = os.getenv("ALPACA_API_KEY_ID")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://data.alpaca.markets")

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
}

def alpaca_price(symbol: str) -> float:
    """Letzter Kurs (Quote) eines Symbols"""
    url = f"{BASE_URL}/v2/stocks/{symbol}/quotes/latest"
    r = requests.get(url, headers=HEADERS, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"Alpaca API error {r.status_code}: {r.text}")
    data = r.json()
    return data["quote"]["ap"]

def alpaca_history(symbol: str, days=180) -> pd.Series:
    """Tagesbalken der letzten n Tage"""
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    url = f"{BASE_URL}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": "1Day",
        "start": start.isoformat() + "Z",
        "end": end.isoformat() + "Z",
        "limit": days,
    }
    r = requests.get(url, headers=HEADERS, params=params, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"Alpaca API error {r.status_code}: {r.text}")
    bars = r.json().get("bars", [])
    if not bars:
        return pd.Series(dtype=float)
    df = pd.DataFrame(bars)
    df["t"] = pd.to_datetime(df["t"])
    df.set_index("t", inplace=True)
    return df["c"]
