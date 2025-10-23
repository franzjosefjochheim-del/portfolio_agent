# alpaca_data.py
import os, time, random, requests, pandas as pd
from datetime import datetime, timedelta

ALPACA_KEY   = os.getenv("ALPACA_API_KEY_ID", "")
ALPACA_SECRET= os.getenv("ALPACA_API_SECRET_KEY", "")
BASE_URL     = os.getenv("ALPACA_BASE_URL", "https://data.alpaca.markets")

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
    "Accept": "application/json",
}

def _http_get(url, params=None, retries=4, timeout=15):
    backoff = 0.8
    for i in range(retries):
        r = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
        if 200 <= r.status_code < 300:
            return r.json()
        # 429/5xx → Backoff + Retry
        if r.status_code in (429, 500, 502, 503, 504):
            ra = r.headers.get("Retry-After")
            sleep_for = float(ra) if ra and ra.isdigit() else backoff + random.random()*0.5
            time.sleep(sleep_for)
            backoff = min(backoff * 2, 6.0)
            continue
        r.raise_for_status()
    raise RuntimeError(f"Alpaca HTTP {r.status_code}: {r.text}")

# ---------- STOCKS / ETFs (US) ----------
def is_us_symbol(symbol: str) -> bool:
    # Sehr einfache Heuristik: keine Suffixe wie .DE, .PA, .HK, .AX etc.
    return "." not in symbol and ":" not in symbol and "-" not in symbol and symbol.isupper()

def alpaca_price_stock(symbol: str) -> float:
    """
    Letzter Ask-Preis ('ap') über Snapshot-Quote.
    Doku: GET /v2/stocks/{symbol}/quotes/latest
    """
    url = f"{BASE_URL}/v2/stocks/{symbol}/quotes/latest"
    data = _http_get(url)
    return float(data["quote"]["ap"])

def alpaca_history_stock(symbol: str, days=180) -> pd.Series:
    """
    Tagesbars über /v2/stocks/{symbol}/bars (timeframe=1Day)
    """
    end = datetime.utcnow()
    start = end - timedelta(days=days + 10)
    url = f"{BASE_URL}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": "1Day",
        "start": start.isoformat(timespec="seconds") + "Z",
        "end":   end.isoformat(timespec="seconds") + "Z",
        "limit": days + 10,
        "adjustment": "all",
    }
    data = _http_get(url, params=params)
    bars = data.get("bars", [])
    if not bars: 
        return pd.Series(dtype=float)
    df = pd.DataFrame(bars)
    df["t"] = pd.to_datetime(df["t"])
    df.set_index("t", inplace=True)
    s = df["c"]
    s.index = s.index.tz_localize("UTC").tz_convert("Europe/Berlin")
    return s.asfreq("D").last("180D").dropna()

# ---------- CRYPTO (über Alpaca v1beta3) ----------
# Wir nehmen USD-Paare; dein Portfolio rechnet ohnehin in EUR → Umrechnung macht der Rest deines Codes
CRYPTO_BASE = f"{BASE_URL}/v1beta3/crypto/us"

def _to_pair(coin_id: str) -> str:
    # einfache Map: "bitcoin" → "BTC/USD", "ethereum" → "ETH/USD", "solana" → "SOL/USD", "ripple" → "XRP/USD", "cardano" → "ADA/USD", "chainlink" → "LINK/USD"
    m = {
        "bitcoin":"BTC/USD","ethereum":"ETH/USD","solana":"SOL/USD","ripple":"XRP/USD",
        "xrp":"XRP/USD","cardano":"ADA/USD","ada":"ADA/USD","chainlink":"LINK/USD","link":"LINK/USD"
    }
    if coin_id.upper().endswith("/USD"): return coin_id.upper()
    return m.get(coin_id.lower(), f"{coin_id.upper()}/USD")

def alpaca_price_crypto(coin_id: str) -> float:
    """
    Letzter Preis aus Quote (ask) für Crypto-Paar.
    Doku: GET /v1beta3/crypto/us/{symbol}/quotes/latest
    """
    pair = _to_pair(coin_id)
    url  = f"{CRYPTO_BASE}/{pair}/quotes/latest"
    data = _http_get(url)
    return float(data["quote"]["ap"])

def alpaca_history_crypto(coin_id: str, days=180) -> pd.Series:
    """
    Tagesbars (1Day) für Crypto-Paar.
    Doku: GET /v1beta3/crypto/us/{symbol}/bars
    """
    pair = _to_pair(coin_id)
    end = datetime.utcnow()
    start = end - timedelta(days=days + 10)
    url = f"{CRYPTO_BASE}/{pair}/bars"
    params = {
        "timeframe": "1Day",
        "start": start.isoformat(timespec="seconds") + "Z",
        "end":   end.isoformat(timespec="seconds") + "Z",
        "limit": days + 10,
    }
    data = _http_get(url, params=params)
    bars = data.get("bars", [])
    if not bars:
        return pd.Series(dtype=float)
    df = pd.DataFrame(bars)
    df["t"] = pd.to_datetime(df["t"])
    df.set_index("t", inplace=True)
    s = df["c"]
    s.index = s.index.tz_localize("UTC").tz_convert("Europe/Berlin")
    return s.asfreq("D").last("180D").dropna()
