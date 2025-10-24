# monitor.py — Portfolio-Agent (Render, quiet logs edition)
#
# Änderungen ggü. letzter Version:
# - CoinGecko 429 wird NICHT mehr geloggt (kein roter Spam alle 15min)
# - yfinance "Failed to get ticker ... possibly delisted" wird unterdrückt
#   durch yfinance-Logger auf CRITICAL und kein eigenes print bei normalem Miss
# - Nur noch wirklich wichtige Prints (Start, Scheduler, Alerts/Reports)
#
# Features bleiben:
# - Tagesbericht 06:30
# - Intraday-Überwachung 06:10–22:00 (Mo–Fr)
# - Chancenfinder 4×/Tag
# - Trailing-Stop / Take-Profit Checks
# - Auto-Watchlist US-only Stocks + deine Kryptos
# - Telegram-Push
# - Alpaca bevorzugt für US-Preise, Fallback yfinance
# - CoinGecko gecacht, damit wir Portfolio-Wert auch bei Rate Limit weiterrechnen können

import os, json, time, math, re, logging
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import yfinance as yf
import numpy as np
import pandas as pd

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# yfinance logger stummschalten, sonst spammt es "possibly delisted"
yf_logger = logging.getLogger("yfinance")
yf_logger.setLevel(logging.CRITICAL)

# ----------------------------- Konfiguration -----------------------------

TZ = ZoneInfo("Europe/Berlin")
STATE_FILE = "last_state.json"
WATCHLIST_FILE = "watchlist.json"

MIN_ALERT_EUR = float(os.getenv("MIN_ALERT_EUR", "500"))
MAX_JUMP_PCT  = float(os.getenv("MAX_JUMP_PCT", "50"))

ASSET_MOVE_PCT = float(os.getenv("ASSET_MOVE_PCT", "2"))
PORT_MOVE_PCT  = float(os.getenv("PORT_MOVE_PCT", "1"))

# CoinGecko Rate Limit Handling
CG_MIN_INTERVAL_SEC = float(os.getenv("CG_MIN_INTERVAL_SEC", "20"))
CG_GLOBAL_GRACE_SEC = 30.0  # nach Prozessstart erstmal nix hammern
_PROCESS_START_TS = time.time()

_last_cg_call = {}        # pro Coin, wann zuletzt live abgefragt
_cg_price_cache = {}      # letzter bekannter Preis pro Coin
_cg_history_cache = {}    # letzter bekannter History-TS pro Coin (Series)

# yfinance Call-Pause zwischen tickern (ms)
YF_PAUSE_MS = int(os.getenv("YF_PAUSE_MS", "400"))

# Alpaca Zugang
ALPACA_ENABLED = os.getenv("ALPACA_ENABLED", "false").lower() in ("1","true","yes","y")
APCA_API_BASE_URL = os.getenv("APCA_API_BASE_URL", "").rstrip("/")
APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")

# Wir ignorieren absichtlich London/HK/Paris Ticker usw.
US_LIVE_TICKERS_ONLY = True

BASE_UNIVERSE_STOCKS = [
    "NVDA",
    "KO",
    "OXY",
    "FVRR",
    "IYF"
]

BASE_UNIVERSE_CRYPTO = [
    "bitcoin", "ethereum", "solana", "ripple", "cardano", "chainlink"
]

def now_str():
    return datetime.now(TZ).strftime("%d.%m.%Y %H:%M:%S")

def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _norm_code(x: str) -> str:
    return re.sub(r"\s+", "", str(x or "")).upper()

def _dedup_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        k = _norm_code(x)
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out

def _sleep_ms(ms):
    if ms > 0:
        time.sleep(ms / 1000.0)

# ----------------------------- Preisquellen Aktien ---------------------

def alpaca_last_price(ticker: str) -> float:
    """
    Versucht Alpaca-Quote (oder letzter Trade). Wenn kein Preis -> NaN.
    Wir loggen KEINE Warns hier, außer echter Exception.
    """
    if not (ALPACA_ENABLED and APCA_API_BASE_URL and APCA_API_KEY_ID and APCA_API_SECRET_KEY):
        return float("nan")
    if "." in ticker:
        # Alpaca kann z.B. 'VUSD.L' eh nicht
        return float("nan")
    headers = {
        "APCA-API-KEY-ID": APCA_API_KEY_ID,
        "APCA-API-SECRET-KEY": APCA_API_SECRET_KEY,
    }
    try:
        # Quotes
        url = f"{APCA_API_BASE_URL}/v2/stocks/{ticker}/quotes/latest"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            bp = data.get("quote", {}).get("bp")
            ap = data.get("quote", {}).get("ap")
            if bp is not None and ap is not None and ap > 0:
                return float((bp + ap) / 2.0)

        # Trades
        url = f"{APCA_API_BASE_URL}/v2/stocks/{ticker}/trades/latest"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            px = data.get("trade", {}).get("p")
            if px is not None:
                return float(px)

        # Wenn Alpaca den Ticker nicht kennt => still
        return float("nan")
    except Exception:
        # echte Netzwerk/JSON Fehler -> einmal kurze Debug-Ausgabe
        print(f"[{now_str()}] DBG alpaca exception for {ticker}")
        return float("nan")

def yf_price_us_only(ticker: str) -> float:
    """
    Holt letzten Preis aus yfinance (US only).
    Keine Warn-Prints bei "kein Ticker".
    """
    if "." in ticker:
        return float("nan")
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None)
        if fi and getattr(fi, "last_price", None):
            return float(fi.last_price)
        hist = t.history(period="1d", interval="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        # yfinance kann hart failen z.B. Rate Limit -> still
        return float("nan")
    return float("nan")

def yf_history_us_only(ticker: str, period="6mo", interval="1d") -> pd.Series:
    """
    Für Scanner (RSI/MACD). Wenn nix kommt: leere Series,
    kein lautstarkes Logging.
    """
    if "." in ticker:
        return pd.Series(dtype=float)
    try:
        t = yf.Ticker(ticker)
        h = t.history(period=period, interval=interval)
        if not h.empty:
            return h["Close"].dropna()
    except Exception:
        return pd.Series(dtype=float)
    return pd.Series(dtype=float)

def get_stock_price(ticker: str) -> float:
    """
    Priorität:
    1. Alpaca
    2. yfinance
    Ergebnis NaN falls nix.
    """
    px = alpaca_last_price(ticker)
    if px == px and px > 0:
        return px
    _sleep_ms(YF_PAUSE_MS)
    px = yf_price_us_only(ticker)
    if px == px and px > 0:
        return px
    return float("nan")

# ----------------------------- CoinGecko Handling ----------------------

def _load_state_for_crypto_cache():
    """
    Holt alte Preise aus STATE_FILE (damit wir bei 429 noch was haben).
    """
    st = load_json(STATE_FILE, {"crypto_cache": {}})
    return st.get("crypto_cache", {})

def _save_state_crypto_cache():
    """
    Speichert _cg_price_cache zurück in STATE_FILE.
    """
    st = load_json(STATE_FILE, {})
    if "crypto_cache" not in st:
        st["crypto_cache"] = {}
    for sym, val in _cg_price_cache.items():
        st["crypto_cache"][sym] = {
            "price": val.get("price"),
            "ts": val.get("ts")
        }
    save_json(STATE_FILE, st)

def _cg_allowed_now(coin_id: str) -> bool:
    """
    Drossel: CoinGecko nur alle CG_MIN_INTERVAL_SEC pro Coin
    + in den ersten CG_GLOBAL_GRACE_SEC nach Start alles unterdrücken,
      damit beim Deploy kein Burst kommt.
    """
    now_ts = time.time()
    if now_ts - _PROCESS_START_TS < CG_GLOBAL_GRACE_SEC:
        return False
    last = _last_cg_call.get(coin_id, 0)
    if now_ts - last < CG_MIN_INTERVAL_SEC:
        return False
    return True

def coingecko_price(coin_id: str) -> float:
    """
    Preis in EUR pro Coin.
    Strategie:
      - Wenn Cache vorhanden, gib Cache
      - Wenn wir noch nicht wieder fragen dürfen: versuch STATE-Fallback
      - Wenn wir fragen dürfen: Call -> Cache -> STATE
      - Bei Fehler/429: KEIN print (wir bleiben still), nur Fallback
    """
    # 1. lokaler RAM-Cache first
    if coin_id in _cg_price_cache and "price" in _cg_price_cache[coin_id]:
        return float(_cg_price_cache[coin_id]["price"])

    # 2. dürfen wir nicht? -> STATE fallback
    if not _cg_allowed_now(coin_id):
        old_prices = _load_state_for_crypto_cache()
        if coin_id in old_prices and "price" in old_prices[coin_id]:
            return float(old_prices[coin_id]["price"])
        return float("nan")

    # 3. echter Call
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        r = requests.get(url, params={"ids": coin_id, "vs_currencies": "eur"}, timeout=20)
        r.raise_for_status()
        px = float(r.json()[coin_id]["eur"])
        _last_cg_call[coin_id] = time.time()
        _cg_price_cache[coin_id] = {"price": px, "ts": _last_cg_call[coin_id]}
        _save_state_crypto_cache()
        return px
    except Exception:
        # CoinGecko 429 oder Netzfehler -> still fallback
        old_prices = _load_state_for_crypto_cache()
        if coin_id in old_prices and "price" in old_prices[coin_id]:
            return float(old_prices[coin_id]["price"])
        return float("nan")

def coingecko_history(coin_id: str, days=180) -> pd.Series:
    """
    Historie für RSI/MACD etc.
    - Wenn Cache vorhanden: nimm Cache.
    - Wenn wir nicht dürfen: leere Series (Scanner kriegt dann einfach kein Setup).
    - Bei Fehler/429: leere Series (still).
    """
    if coin_id in _cg_history_cache and not _cg_history_cache[coin_id].empty:
        return _cg_history_cache[coin_id]

    if not _cg_allowed_now(coin_id):
        return pd.Series(dtype=float)

    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        r = requests.get(url, params={"vs_currency": "eur", "days": days}, timeout=30)
        r.raise_for_status()
        prices = r.json().get("prices", [])
        if not prices:
            return pd.Series(dtype=float)

        ts = [pd.to_datetime(p[0], unit="ms") for p in prices]
        vals = [float(p[1]) for p in prices]
        s = pd.Series(vals, index=pd.DatetimeIndex(ts, tz="UTC")).tz_convert(TZ)
        s = s.resample("1D").last().dropna()

        _last_cg_call[coin_id] = time.time()
        _cg_history_cache[coin_id] = s
        return s
    except Exception:
        return pd.Series(dtype=float)

def get_crypto_price(symbol: str) -> float:
    return coingecko_price(symbol)

# ----------------------------- Benachrichtigungen ----------------------

def send_telegram(msg: str):
    token = os.getenv("TG_BOT_TOKEN")
    chat_id = os.getenv("TG_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": msg,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        requests.post(url, data=payload, timeout=15)
    except Exception as e:
        # Telegram sollte eigentlich laufen; falls nicht, EINmal laut sagen
        print(f"[{now_str()}] WARN telegram: {e}")

def notify(title: str, text: str):
    """
    Das ist der EINZIGE "laute" Kanal.
    Alles Wichtige (Alarme, Tagesbericht, Scanner-Ergebnis)
    geht über print() UND Telegram.
    """
    header = f"📣 {title}\n{now_str()}\n\n{text}"
    print(header)
    send_telegram(header)

# ----------------------------- Portfolio-Core -------------------------

def load_positions() -> dict:
    p = load_json("positions.json", {})
    p.setdefault("stocks", [])
    p.setdefault("crypto", [])
    return p

def _compute_portfolio_raw():
    pos = load_positions()
    items = []

    # Stocks/ETFs
    for s in pos["stocks"]:
        ticker = s.get("ticker", "")
        qty = float(s.get("quantity", 0))

        px = get_stock_price(ticker)
        val = px * qty if px == px else float("nan")

        items.append({
            "kind": "stock",
            "name": s["name"],
            "code": ticker,
            "qty": qty,
            "price": px,
            "value": val,
            "sl_pct": float(s.get("stop_loss_pct", 0)),
            "tp_pct": float(s.get("take_profit_pct", 0))
        })

    # Crypto
    for c in pos["crypto"]:
        sym = c.get("symbol", "")
        qty = float(c.get("quantity", 0))

        px = get_crypto_price(sym)
        val = px * qty if px == px else float("nan")

        items.append({
            "kind": "crypto",
            "name": c["name"],
            "code": sym,
            "qty": qty,
            "price": px,
            "value": val,
            "sl_pct": float(c.get("stop_loss_pct", 0)),
            "tp_pct": float(c.get("take_profit_pct", 0))
        })

    total = sum(v["value"] for v in items if v["value"] == v["value"])
    return items, total

def compute_portfolio():
    """
    Portfolio mit Failover:
    - wenn total==0 oder Sprung > MAX_JUMP_PCT: einmal 2s warten & Retry
    """
    items, total = _compute_portfolio_raw()

    state = load_json(STATE_FILE, {"portfolio": {"total": None}})
    old_total = state["portfolio"].get("total")
    jump_ok = True
    if old_total and old_total > 0 and total > 0:
        d = abs((total - old_total) / old_total * 100.0)
        jump_ok = d <= MAX_JUMP_PCT

    if (total == 0) or (not jump_ok):
        time.sleep(2.0)
        items2, total2 = _compute_portfolio_raw()
        if total2 > 0:
            items, total = items2, total2

    return items, total

def render_report(items, total):
    lines = [f"📊 Portfolio-Report — {now_str()}",
             "---------------------------------------------"]
    for it in items:
        px = it["price"]
        pxs = "n/a" if not (px == px) else f"{px:,.2f} €"
        val = it["value"]
        vals = "n/a" if not (val == val) else f"{val:,.2f} €"
        lines.append(f"{it['name']} ({it['code']})  {it['qty']:.6f} × {pxs} = {vals}")
    lines.append("---------------------------------------------")
    lines.append(f"💰 Gesamtwert: {total:,.2f} €")
    return "\n".join(lines)

# ----------------------------- Alerts / Risk --------------------------

def pct_change(new, old):
    if old is None or old == 0 or not (new == new) or not (old == old):
        return None
    return (new - old) / old * 100.0

def _top_movers(items, old_state, n=3):
    movers = []
    for it in items:
        code = it["code"] or it["name"]
        px = it["price"]
        st = old_state.get(code, {})
        old_px = st.get("price")
        chg = pct_change(px, old_px)
        if chg is not None:
            movers.append((chg, it["name"], code, px))
    gainers = sorted([m for m in movers if m[0] > 0],
                     key=lambda x: x[0], reverse=True)[:n]
    losers  = sorted([m for m in movers if m[0] < 0],
                     key=lambda x: x[0])[:n]
    return gainers, losers

def evaluate_alerts(items, total):
    state = load_json(STATE_FILE, {"portfolio": {"total": None}, "assets": {}, "crypto_cache": {}})
    alerts = []

    old_total = state["portfolio"].get("total")

    if (old_total is not None and
        old_total >= MIN_ALERT_EUR and
        total      >= MIN_ALERT_EUR):

        d_port = pct_change(total, old_total)
        if d_port is not None and abs(d_port) >= PORT_MOVE_PCT:
            gainers, losers = _top_movers(items, state.get("assets", {}), n=3)
            lines = [f"📈/📉 Portfolio-Bewegung: {d_port:+.2f}%  "
                     f"(neu {total:,.2f} €; alt {old_total:,.2f} €)"]
            if gainers:
                gtxt = "; ".join([
                    f"{nm} ({cd}) {chg:+.2f}% @ {px:,.2f}€"
                    for chg, nm, cd, px in gainers
                ])
                lines.append(f"Top-Gewinner: {gtxt}")
            if losers:
                ltxt = "; ".join([
                    f"{nm} ({cd}) {chg:+.2f}% @ {px:,.2f}€"
                    for chg, nm, cd, px in losers
                ])
                lines.append(f"Top-Verlierer: {ltxt}")
            alerts.append("\n".join(lines))

    new_assets = {}
    for it in items:
        code = it["code"] or it["name"]
        px = it["price"]
        st = state["assets"].get(code, {"price": None, "high": None, "low": None})

        d = pct_change(px, st["price"])
        if d is not None and abs(d) >= ASSET_MOVE_PCT:
            alerts.append(f"• {it['name']} ({code}) bewegt sich {d:+.2f}%  "
                          f"(Preis {px:,.2f} €)")

        high = px if st["high"] is None else max(st["high"], px)
        low  = px if st["low"]  is None else min(st["low"],  px)

        sl_pct = it.get("sl_pct", 0.0)
        tp_pct = it.get("tp_pct", 0.0)

        if sl_pct > 0 and high == high and px == px:
            sl_level = high * (1 - sl_pct/100.0)
            if px <= sl_level:
                alerts.append(
                    f"⛔ Trailing-Stop erreicht: {it['name']} ({code}) — "
                    f"Preis {px:,.2f} € ≤ SL {sl_level:,.2f} € "
                    f"(−{sl_pct:.1f}% vom High {high:,.2f} €)"
                )
                new_assets[code] = {
                    "price": px, "high": high, "low": low,
                    "need_redeploy": True
                }

        if tp_pct > 0 and low == low and px == px:
            tp_level = low * (1 + tp_pct/100.0)
            if px >= tp_level:
                alerts.append(
                    f"✅ Take-Profit erreicht: {it['name']} ({code}) — "
                    f"Preis {px:,.2f} € ≥ TP {tp_level:,.2f} € "
                    f"(+{tp_pct:.1f}% vom Low {low:,.2f} €)"
                )
                new_assets[code] = {
                    "price": px, "high": high, "low": low,
                    "need_redeploy": True
                }

        if code not in new_assets:
            new_assets[code] = {"price": px, "high": high, "low": low}

    save_json(
        STATE_FILE,
        {
            "portfolio": {"total": total},
            "assets": new_assets,
            "crypto_cache": load_json(STATE_FILE, {}).get("crypto_cache", {})
        }
    )
    return alerts

# ----------------------------- TA Helpers -----------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    ma_up = up.ewm(com=period-1, adjust=False).mean()
    ma_down = down.ewm(com=period-1, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi_v = 100 - (100 / (1 + rs))
    return rsi_v

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

# ----------------------------- Watchlist ------------------------------

def sync_watchlist_from_positions(max_stocks=None, max_crypto=None):
    """
    Baut watchlist.json neu anhand deines echten Depots + Basisuniversum (US only).
    London/HK/Paris-Ticker werden gefiltert raus.
    """
    max_stocks = int(os.getenv("MAX_STOCKS", str(max_stocks if max_stocks else 30)))
    max_crypto = int(os.getenv("MAX_CRYPTO", str(max_crypto if max_crypto else 10)))

    pos = load_positions()

    pos_stocks = []
    for s in pos.get("stocks", []):
        t = _norm_code(s.get("ticker", ""))
        if t and "." not in t:
            pos_stocks.append(t)

    pos_crypto = [_norm_code(c.get("symbol","")) for c in pos.get("crypto", []) if c.get("symbol")]

    pool_stocks = pos_stocks + BASE_UNIVERSE_STOCKS
    pool_crypto = pos_crypto + BASE_UNIVERSE_CRYPTO

    stocks_final = _dedup_keep_order([t for t in pool_stocks if "." not in t])[:max_stocks]
    crypto_final = _dedup_keep_order(pool_crypto)[:max_crypto]

    wl = load_json(WATCHLIST_FILE, {"stocks": [], "crypto": []})
    wl["stocks"] = _dedup_keep_order(stocks_final + wl.get("stocks", []))
    wl["stocks"] = [t for t in wl["stocks"] if "." not in t][:max_stocks]
    wl["crypto"] = _dedup_keep_order(crypto_final + wl.get("crypto", []))[:max_crypto]

    save_json(WATCHLIST_FILE, wl)
    return wl

def load_watchlist():
    return sync_watchlist_from_positions()

# ----------------------------- Scanner --------------------------------

def score_stock(ticker: str) -> dict | None:
    close = yf_history_us_only(ticker, period="6mo", interval="1d")
    if close.empty or len(close) < 50:
        return None

    last = float(close.iloc[-1])
    rsi14 = float(rsi(close, 14).iloc[-1])

    macd_line, signal_line, hist = macd(close)
    macd_last    = float(macd_line.iloc[-1])
    signal_last  = float(signal_line.iloc[-1])
    hist_last    = float(hist.iloc[-1])

    sma20  = sma(close, 20)
    sma50  = sma(close, 50)
    sma200 = sma(close, 200)

    sma20_l  = float(sma20.iloc[-1])  if not np.isnan(sma20.iloc[-1])  else float("nan")
    sma50_l  = float(sma50.iloc[-1])  if not np.isnan(sma50.iloc[-1])  else float("nan")
    sma200_l = float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else float("nan")

    ret_7d  = (last / float(close.iloc[-7])  - 1) * 100 if len(close) >= 8  else 0.0
    ret_30d = (last / float(close.iloc[-30]) - 1) * 100 if len(close) >= 31 else 0.0

    score = 0
    notes = []

    if not math.isnan(sma200_l) and (last > sma50_l > sma200_l):
        score += 2; notes.append("Trend ↑ (Preis>SMA50>SMA200)")
    elif not math.isnan(sma50_l) and last > sma50_l:
        score += 1; notes.append("Trend ↑ (Preis>SMA50)")

    if 35 <= rsi14 <= 60:
        score += 1; notes.append(f"RSI {rsi14:.0f}")
    elif rsi14 < 30:
        score += 1; notes.append(f"RSI {rsi14:.0f} (überverkauft)")

    if macd_last > signal_last and hist_last > 0:
        score += 1; notes.append("MACD bullisch")

    if ret_7d  > 2: score += 1; notes.append(f"7d {ret_7d:+.1f}%")
    if ret_30d > 4: score += 1; notes.append(f"30d {ret_30d:+.1f}%")

    entry_hint = None
    if not math.isnan(sma20_l):
        dist = (last / sma20_l - 1) * 100
        if abs(dist) <= 2:
            entry_hint = f"Nahe SMA20 (Δ {dist:+.1f}%)"
            score += 1

    return {
        "code": ticker,
        "price": last,
        "score": score,
        "notes": ", ".join(notes),
        "rsi": rsi14,
        "macd": macd_last - signal_last,
        "ret7": ret_7d,
        "ret30": ret_30d,
        "entry_hint": entry_hint
    }

def score_crypto(coin_id: str) -> dict | None:
    close = coingecko_history(coin_id, days=180)
    if close.empty or len(close) < 50:
        return None

    last  = float(close.iloc[-1])
    rsi14 = float(rsi(close, 14).iloc[-1])

    macd_line, signal_line, hist = macd(close)
    macd_last   = float(macd_line.iloc[-1])
    signal_last = float(signal_line.iloc[-1])
    hist_last   = float(hist.iloc[-1])

    sma20  = sma(close, 20)
    sma50  = sma(close, 50)
    sma200 = sma(close, 200)

    sma20_l  = float(sma20.iloc[-1])
    sma50_l  = float(sma50.iloc[-1])
    sma200_l = float(sma200.iloc[-1])

    ret_7d  = (last / float(close.iloc[-7])  - 1) * 100 if len(close) >= 8  else 0.0
    ret_30d = (last / float(close.iloc[-30]) - 1) * 100 if len(close) >= 31 else 0.0

    score = 0
    notes = []

    if last > sma50_l > sma200_l:
        score += 2; notes.append("Trend ↑ (Preis>SMA50>SMA200)")
    elif last > sma50_l:
        score += 1; notes.append("Trend ↑ (Preis>SMA50)")

    if 35 <= rsi14 <= 60:
        score += 1; notes.append(f"RSI {rsi14:.0f}")
    elif rsi14 < 30:
        score += 1; notes.append(f"RSI {rsi14:.0f} (überverkauft)")

    if macd_last > signal_last and hist_last > 0:
        score += 1; notes.append("MACD bullisch")

    if ret_7d  > 3: score += 1; notes.append(f"7d {ret_7d:+.1f}%")
    if ret_30d > 6: score += 1; notes.append(f"30d {ret_30d:+.1f}%")

    entry_hint = None
    dist = (last / sma20_l - 1) * 100 if sma20_l else float("nan")
    if not math.isnan(dist) and abs(dist) <= 2.5:
        entry_hint = f"Nahe SMA20 (Δ {dist:+.1f}%)"
        score += 1

    return {
        "code": coin_id,
        "price": last,
        "score": score,
        "notes": ", ".join(notes),
        "rsi": rsi14,
        "macd": macd_last - signal_last,
        "ret7": ret_7d,
        "ret30": ret_30d,
        "entry_hint": entry_hint
    }

def suggest_position_sizes(total_value: float, risk_level: str = "balanced"):
    pct_map = {"conservative": 0.02, "balanced": 0.03, "aggressive": 0.05}
    base_pct = pct_map.get(risk_level, 0.03)
    per_trade_eur = total_value * base_pct
    min_depot = 1500.0
    safety_buffer = max(total_value - min_depot, 0)
    per_trade_eur = min(per_trade_eur, safety_buffer * 0.2)
    return round(per_trade_eur, 2)

def job_scanner():
    wl = load_watchlist()
    items, total = compute_portfolio()
    if total <= 0:
        return

    scored_stocks = []
    scored_crypto = []

    for t in wl["stocks"]:
        res = score_stock(t)
        if res:
            scored_stocks.append(res)

    for c in wl["crypto"]:
        res = score_crypto(c)
        if res:
            scored_crypto.append(res)

    top_s = sorted(scored_stocks, key=lambda x: x["score"], reverse=True)[:3]
    top_c = sorted(scored_crypto, key=lambda x: x["score"], reverse=True)[:3]

    if not top_s and not top_c:
        notify("Chancenfinder", "Keine klaren Setups aktuell – Kapital in Reserve halten (Schutzmodus).")
        return

    risk_mode = os.getenv("RISK_MODE", "balanced")
    size_eur = suggest_position_sizes(total, risk_mode)

    def fmt(entry):
        hint = f" • {entry['entry_hint']}" if entry.get("entry_hint") else ""
        return (f"{entry['code']}: Score {entry['score']} | Preis ~{entry['price']:,.2f} € | "
                f"RSI {entry['rsi']:.0f}, MACDΔ {entry['macd']:+.3f}, "
                f"7d {entry['ret7']:+.1f}%, 30d {entry['ret30']:+.1f}%"
                f"{hint} — Vorschlag: ~{size_eur:,.2f} €")

    lines = []
    if top_s:
        lines.append("📈 Aktien/ETFs – Top Setups:")
        lines += ["• " + fmt(e) for e in top_s]
    if top_c:
        lines.append("\n🪙 Krypto – Top Setups:")
        lines += ["• " + fmt(e) for e in top_c]

    notify("Chancenfinder – neue Reinvest-Ideen", "\n".join(lines))

# ----------------------------- Jobs ----------------------------------

def job_daily_summary():
    items, total = compute_portfolio()
    report = render_report(items, total)
    notify("Tagesbericht", report)

def job_intraday_monitor():
    """
    Alle 15min (Mo-Fr 06:10–22:00) → prüft Bewegungen/Stops
    Außerhalb der Zeiten: still updaten aber kein Push.
    """
    t = datetime.now(TZ).time()
    if not (t >= datetime.strptime("06:10", "%H:%M").time() and
            t <= datetime.strptime("22:00", "%H:%M").time()):
        items, total = compute_portfolio()
        _ = evaluate_alerts(items, total)
        return

    items, total = compute_portfolio()

    if total < MIN_ALERT_EUR:
        # Nur State refreshen, kein Spam wenn Depot klein ist
        state = load_json(STATE_FILE, {"portfolio": {"total": None}, "assets": {}, "crypto_cache": {}})
        save_json(
            STATE_FILE,
            {
                "portfolio": {"total": total},
                "assets": state.get("assets", {}),
                "crypto_cache": state.get("crypto_cache", {})
            }
        )
        return

    alerts = evaluate_alerts(items, total)
    if alerts:
        notify("Intraday-Alarm", "\n".join(alerts))

def main():
    print(f"[{now_str()}] Agent gestartet. Alpaca={'on' if ALPACA_ENABLED else 'off'}")

    # Crypto-Cache aus STATE_FILE reinziehen (damit wir sofort Preise haben ohne CG-Hit)
    old_crypto_cache = _load_state_for_crypto_cache()
    for sym, d in old_crypto_cache.items():
        if "price" in d:
            _cg_price_cache[sym] = {"price": d["price"], "ts": d.get("ts", time.time())}

    # Watchlist syncen (filtert automatisch nur US-stocks u. deine coins)
    sync_watchlist_from_positions()

    # STATE_FILE initialisieren falls nicht vorhanden (nur 1x nach neuem Deploy)
    if not os.path.exists(STATE_FILE):
        items, total = compute_portfolio()
        save_json(
            STATE_FILE,
            {
                "portfolio": {"total": total},
                "assets": {
                    (it['code'] or it['name']): {
                        "price": it["price"],
                        "high": it["price"],
                        "low":  it["price"]
                    }
                    for it in items
                },
                "crypto_cache": {
                    sym: {"price": dat["price"], "ts": dat["ts"]}
                    for sym, dat in _cg_price_cache.items()
                }
            }
        )
        print(f"[{now_str()}] Initialer Zustand gespeichert.")

    # Scheduler konfigurieren
    sched = BlockingScheduler(timezone=TZ)

    # 06:30 täglicher Report
    sched.add_job(job_daily_summary, CronTrigger(hour=6, minute=30))

    # Intraday-Checks alle 15 Min (Mo–Fr)
    sched.add_job(job_intraday_monitor, CronTrigger(day_of_week='mon-fri', hour='6-22', minute='*/15'))

    # Chancenfinder 4×/Tag
    for hh, mm in [(7,15),(12,15),(16,15),(20,15)]:
        sched.add_job(job_scanner, CronTrigger(day_of_week='mon-sun', hour=hh, minute=mm))

    print(f"[{now_str()}] Scheduler aktiv. Warte auf Jobs …")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print(f"[{now_str()}] Agent gestoppt.")

if __name__ == "__main__":
    main()
