# monitor.py  — Portfolio-Agent (Render)
# -------------------------------------------------------------------
# Was macht der Agent?
# - Tagesbericht morgens (06:30 Europe/Berlin)
# - Intraday-Überwachung alle 15 Min (Mo–Fr 06:10–22:00)
#   -> Asset-Alarm (einzelne Titel bewegen sich stark)
#   -> Portfolio-Alarm (Depot in Summe bewegt sich stark)
#   -> Trailing-Stop / Take-Profit Warnungen
#   -> nennt Top-Mover (mit Titel/Position)
# - Chancenfinder (4x/Tag) scannt Watchlist nach neuen Setups
#
# Benachrichtigung aktuell NUR via Telegram (E-Mail ist deaktiviert).
#
# Datenquellen:
# - Alpaca (Echtzeit US-Ticker wie NVDA, KO, AAPL…)
# - Yahoo Finance (yfinance) als Fallback, vor allem für London-ETFs
# - CoinGecko (Krypto) mit Drosselung, damit kein 429-Spam kommt
#
# Log-Noise reduziert:
# - Wir fragen Alpaca nur bei US-Tickern (NVDA, KO, IYF etc.).
#   London/Paris/HK/... werden gar nicht erst an Alpaca geschickt.
# - yfinance-Fehler/429 werden nur einmal kurz geloggt, nicht gespammt.
#
# Universe verschlankt:
# - Fokus auf London-ETFs & deine echten Kern-US-Titel, kein Asien/Australien mehr
#   => weniger exotische Ticker => weniger "not found"-Müll in den Logs
#
# Dieses File kannst du 1:1 in Render hochladen/ersetzen.
# -------------------------------------------------------------------

import os, json, time, math, re
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import yfinance as yf
import numpy as np
import pandas as pd

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# ----------------------------- Konfiguration / Env -----------------------------

TZ = ZoneInfo("Europe/Berlin")

STATE_FILE = "last_state.json"      # laufender Zustand (für Intraday-Delta)
WATCHLIST_FILE = "watchlist.json"   # dynamische Watchlist
POSITIONS_FILE = "positions.json"   # dein Depot (s. unten neue Version)

# False-Alarm-/Robustheits-Parameter
MIN_ALERT_EUR = float(os.getenv("MIN_ALERT_EUR", "500"))   # Mindest-Portfoliowert für Alerts
MAX_JUMP_PCT  = float(os.getenv("MAX_JUMP_PCT", "50"))     # wenn Sprung >50%, retry 1x

# Limits für Bewegungs-Alerts
ASSET_MOVE_PCT = float(os.getenv("ASSET_MOVE_PCT", "2"))   # pro Asset
PORT_MOVE_PCT  = float(os.getenv("PORT_MOVE_PCT",  "1"))   # Gesamtdepot

# Trading-Universum / Watchlist-Größe
MAX_STOCKS = int(os.getenv("MAX_STOCKS", "30"))
MAX_CRYPTO = int(os.getenv("MAX_CRYPTO", "10"))

# Scanner-Risikomodus ("conservative", "balanced", "aggressive")
RISK_MODE = os.getenv("RISK_MODE", "balanced")

# Throttling / User-Agent für externe APIs
HTTP_UA = os.getenv(
    "HTTP_UA",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124 Safari/537.36"
)
YF_PAUSE_MS = int(os.getenv("YF_PAUSE_MS", "400"))  # Pause zwischen yfinance Calls (ms)
CG_MIN_INTERVAL_SEC = float(os.getenv("CG_MIN_INTERVAL_SEC", "20"))  # min Abstand CG Calls

# Telegram
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID   = os.getenv("TG_CHAT_ID", "")

# Alpaca
ALPACA_ENABLED    = os.getenv("ALPACA_ENABLED", "false").lower() in ("1", "true", "yes", "on")
ALPACA_BASE_URL   = os.getenv("APCA_API_BASE_URL", "https://data.alpaca.markets")
ALPACA_KEY_ID     = os.getenv("APCA_API_KEY_ID", "")
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")

# ----------------------------- Watchlist-Universum -----------------------------
# Gestrafft: London-ETFs + Kern-US-Tech/Quality + breite US-ETFs
# -> keine Hongkong-, Australien-, Paris-Exoten mehr
BASE_UNIVERSE_STOCKS = [
    # Breite US-Märkte
    "SPY", "QQQ",
    # US Quality / bekannte Large Caps
    "AAPL", "MSFT", "NVDA", "AMD", "KO", "OXY", "IYF",
    # Deine London/IE-basierten ETFs
    "VUSD.L", "VUAA.L", "SWDA.L",
]

# Krypto (CoinGecko IDs)
BASE_UNIVERSE_CRYPTO = [
    "bitcoin",
    "ethereum",
    "solana",
    "cardano",
    "ripple",
    "chainlink",
]

# Optionaler Zusatzkorb (aggressivere Titel, rein wenn EXPAND_UNIVERSE=true)
EXTRA_UNIVERSE_STOCKS = [
    "META", "AMZN", "GOOGL", "TSLA", "AVGO"
]
EXTRA_UNIVERSE_CRYPTO = [
    "polkadot", "uniswap", "litecoin"
]

# ----------------------------- interne Helpers -----------------------------

_last_cg_call = 0.0  # für CoinGecko-Rate-Limit

def now_str() -> str:
    return datetime.now(TZ).strftime("%d.%m.%Y %H:%M:%S")

def _dbg(msg: str):
    # kompaktes Logging mit Timestamp (landet in Render Logs)
    print(f"[{now_str()}] {msg}")

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

# ----------------------------- Telegram Notify -----------------------------

def send_telegram(msg: str):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TG_CHAT_ID,
            "text": msg,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        requests.post(url, data=payload, timeout=15)
    except Exception as e:
        _dbg(f"WARN telegram: {e}")

def notify(title: str, text: str):
    header = f"📣 {title}\n{now_str()}\n\n{text}"
    print(header)
    send_telegram(header)

# ----------------------------- Marktdaten: Alpaca / yfinance / CoinGecko -----

def _looks_like_us_equity(ticker: str) -> bool:
    """
    Heuristik, ob wir Alpaca überhaupt fragen:
    - kein Punkt (also nicht 'VUSD.L')
    - erstes Zeichen keine Zahl (also nicht '0175.HK')
    - Länge max. 5
    - nur A-Z
    Beispiele True: AAPL, NVDA, KO, IYF
    Beispiele False: VUSD.L, SWDA.L, VUAA.L, CHIP.PA, 0175.HK
    """
    if not ticker:
        return False
    if "." in ticker:
        return False
    if ticker[0].isdigit():
        return False
    if len(ticker) > 5:
        return False
    return ticker.isalpha() and ticker.upper() == ticker

def alpaca_price(ticker: str) -> float | None:
    """
    Frage Alpaca nach US-Quote.
    Gibt None zurück, wenn:
    - Alpaca disabled
    - Ticker kein 'US-ähnlicher' Ticker (siehe _looks_like_us_equity)
    - Alpaca kennt den Ticker nicht
    """
    if not ALPACA_ENABLED:
        return None
    if not (ALPACA_KEY_ID and ALPACA_SECRET_KEY):
        return None
    if not _looks_like_us_equity(ticker):
        return None

    try:
        url = f"{ALPACA_BASE_URL}/v2/stocks/{ticker}/quotes/latest"
        headers = {
            "APCA-API-KEY-ID": ALPACA_KEY_ID,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
            "User-Agent": HTTP_UA,
        }
        r = requests.get(url, headers=headers, timeout=5)

        if r.status_code == 404:
            # "gibt's bei Alpaca nicht" -> kein Lärm
            return None

        if r.status_code != 200:
            _dbg(f"WARN alpaca {ticker}: HTTP {r.status_code} {r.text[:120]}")
            return None

        data = r.json()
        quote = data.get("quote", {})
        ap = quote.get("ap")
        bp = quote.get("bp")
        if ap and bp:
            return float((ap + bp) / 2.0)
        if ap:
            return float(ap)
        if bp:
            return float(bp)
        return None

    except Exception as e:
        _dbg(f"WARN alpaca {ticker}: {e}")
        return None

def yf_price(ticker: str) -> float:
    """
    Preis über yfinance, mit kleinem Sleep, ohne Panik bei 429.
    Gibt float('nan') wenn nix geht.
    """
    try:
        time.sleep(YF_PAUSE_MS / 1000.0)
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None)
        if fi and getattr(fi, "last_price", None):
            return float(fi.last_price)

        hist = t.history(period="1d", interval="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])

    except Exception as e:
        _dbg(f"WARN yfinance {ticker}: {e}")

    return float("nan")

def yf_history(ticker: str, period="6mo", interval="1d") -> pd.Series:
    """
    Historische Daten für Scoring (Technische Analyse).
    Für ausländische ETFs klappt das erstaunlich oft.
    """
    try:
        time.sleep(YF_PAUSE_MS / 1000.0)
        t = yf.Ticker(ticker)
        h = t.history(period=period, interval=interval)
        if not h.empty:
            return h["Close"].dropna()
    except Exception as e:
        _dbg(f"WARN yfinance history {ticker}: {e}")
    return pd.Series(dtype=float)

def _respect_cg_rate_limit():
    global _last_cg_call
    elapsed = time.time() - _last_cg_call
    wait_need = CG_MIN_INTERVAL_SEC - elapsed
    if wait_need > 0:
        time.sleep(wait_need)
    _last_cg_call = time.time()

def coingecko_price(coin_id: str) -> float:
    """
    Spotpreis (EUR) von CoinGecko mit Drosselung.
    Gibt nan zurück, wenn CG gerade blockt.
    """
    try:
        _respect_cg_rate_limit()
        url = "https://api.coingecko.com/api/v3/simple/price"
        r = requests.get(
            url,
            params={"ids": coin_id, "vs_currencies": "eur"},
            headers={"User-Agent": HTTP_UA},
            timeout=10,
        )
        if r.status_code != 200:
            _dbg(f"WARN coingecko {coin_id}: HTTP {r.status_code} {r.text[:120]}")
            return float("nan")
        return float(r.json()[coin_id]["eur"])
    except Exception as e:
        _dbg(f"WARN coingecko {coin_id}: {e}")
        return float("nan")

def coingecko_history(coin_id: str, days=180) -> pd.Series:
    """
    Historie (täglich) von CoinGecko für TA-Indikatoren.
    """
    try:
        _respect_cg_rate_limit()
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        r = requests.get(
            url,
            params={"vs_currency": "eur", "days": days},
            headers={"User-Agent": HTTP_UA},
            timeout=10,
        )
        if r.status_code != 200:
            _dbg(f"WARN coingecko hist {coin_id}: HTTP {r.status_code} {r.text[:120]}")
            return pd.Series(dtype=float)

        prices = r.json().get("prices", [])
        if not prices:
            return pd.Series(dtype=float)

        ts = [pd.to_datetime(p[0], unit="ms") for p in prices]
        vals = [float(p[1]) for p in prices]
        s = pd.Series(vals, index=pd.DatetimeIndex(ts, tz="UTC")).tz_convert(TZ)
        s = s.resample("1D").last().dropna()
        return s

    except Exception as e:
        _dbg(f"WARN coingecko history {coin_id}: {e}")
        return pd.Series(dtype=float)

# ----------------------------- Portfolio Core -----------------------------

def load_positions() -> dict:
    """
    Lädt dein Depot.
    Format siehe neue positions.json unten.
    """
    p = load_json(POSITIONS_FILE, {})
    p.setdefault("stocks", [])
    p.setdefault("crypto", [])
    return p

def price_of_stock(ticker: str) -> float:
    """
    Erst Alpaca (falls US/wir haben Glück), sonst yfinance.
    """
    p_alp = alpaca_price(ticker)
    if p_alp is not None:
        return p_alp
    return yf_price(ticker)

def price_of(asset: dict) -> float:
    """
    asset = { "ticker": "...", "quantity": ..., ... }  oder
            { "symbol": "bitcoin", "quantity": ... }
    """
    if "ticker" in asset:
        return price_of_stock(asset["ticker"])
    if "symbol" in asset:
        return coingecko_price(asset["symbol"])
    return float("nan")

def _compute_portfolio_raw():
    """
    Einmalige Berechnung Portfolio, ohne Fail-Safe.
    """
    pos = load_positions()
    items = []

    for s in pos["stocks"]:
        px = price_of(s)
        qty = float(s.get("quantity", 0))
        val = px * qty if px == px else float("nan")
        items.append({
            "kind": "stock",
            "name": s.get("name", s.get("ticker", "")),
            "code": s.get("ticker", ""),
            "qty": qty,
            "price": px,
            "value": val,
            "sl_pct": float(s.get("stop_loss_pct", 0)),
            "tp_pct": float(s.get("take_profit_pct", 0))
        })

    for c in pos["crypto"]:
        px = price_of(c)
        qty = float(c.get("quantity", 0))
        val = px * qty if px == px else float("nan")
        items.append({
            "kind": "crypto",
            "name": c.get("name", c.get("symbol", "")),
            "code": c.get("symbol", ""),
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
    Fail-Safe:
    - wenn total==0 ODER Sprung > MAX_JUMP_PCT gegenüber altem Wert:
      -> 2s warten, nochmal messen, nimm den besseren Versuch.
    """
    items, total = _compute_portfolio_raw()

    state = load_json(STATE_FILE, {"portfolio": {"total": None}})
    old_total = state["portfolio"].get("total")

    jump_ok = True
    if old_total and old_total > 0 and total > 0:
        d = abs((total - old_total) / old_total * 100.0)
        jump_ok = (d <= MAX_JUMP_PCT)

    if (total == 0) or (not jump_ok):
        time.sleep(2.0)
        items2, total2 = _compute_portfolio_raw()
        if total2 > 0:
            items, total = items2, total2

    return items, total

def render_report(items, total):
    lines = [
        f"📊 Portfolio-Report — {now_str()}",
        "---------------------------------------------"
    ]
    for it in items:
        px = it["price"]
        val = it["value"]
        pxs = "n/a" if not (px == px) else f"{px:,.2f} €"
        vals = "n/a" if not (val == val) else f"{val:,.2f} €"

        # Titel/Position (Name + Ticker)
        lines.append(f"{it['name']} ({it['code']})  {it['qty']:.6f} × {pxs} = {vals}")

    lines.append("---------------------------------------------")
    lines.append(f"💰 Gesamtwert: {total:,.2f} €")
    return "\n".join(lines)

# ----------------------------- Risiko / Alerts -----------------------------

def pct_change(new, old):
    if old is None or old == 0 or not (new == new) or not (old == old):
        return None
    return (new - old) / old * 100.0

def _top_movers(items, old_state, n=3):
    """
    Welche Assets sind seit letzter Messung am meisten gestiegen / gefallen
    (basierend auf Preis %Change)?
    """
    movers = []
    for it in items:
        code = it["code"] or it["name"]
        px = it["price"]
        st = old_state.get(code, {})
        old_px = st.get("price")
        chg = pct_change(px, old_px)
        if chg is not None:
            movers.append((chg, it["name"], code, px))

    gainers = sorted(
        [m for m in movers if m[0] > 0],
        key=lambda x: x[0],
        reverse=True
    )[:n]
    losers = sorted(
        [m for m in movers if m[0] < 0],
        key=lambda x: x[0]
    )[:n]

    return gainers, losers

def evaluate_alerts(items, total):
    """
    Prüft:
    - Portfolio Gesamtbewegung
    - Einzelbewegung pro Asset
    - Trailing Stop / Take Profit
    und aktualisiert STATE_FILE.
    """
    state = load_json(STATE_FILE, {"portfolio": {"total": None}, "assets": {}})
    alerts = []

    old_total = state["portfolio"].get("total")

    # Portfolio-Alarm (nur wenn Depot groß genug ist)
    if (old_total is not None and
        old_total >= MIN_ALERT_EUR and
        total     >= MIN_ALERT_EUR):

        d_port = pct_change(total, old_total)
        if d_port is not None and abs(d_port) >= PORT_MOVE_PCT:
            gainers, losers = _top_movers(items, state.get("assets", {}), n=3)

            lines = [
                f"📈/📉 Portfolio-Bewegung: {d_port:+.2f}%  "
                f"(neu {total:,.2f} €; alt {old_total:,.2f} €)"
            ]
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
            alerts.append(
                f"• {it['name']} ({code}) bewegt sich {d:+.2f}% "
                f"(Preis {px:,.2f} €)"
            )

        high = px if st["high"] is None else max(st["high"], px)
        low  = px if st["low"]  is None else min(st["low"],  px)

        sl_pct = it.get("sl_pct", 0.0)
        tp_pct = it.get("tp_pct", 0.0)

        if sl_pct > 0 and high == high and px == px:
            sl_level = high * (1 - sl_pct/100.0)
            if px <= sl_level:
                alerts.append(
                    "⛔ Trailing-Stop erreicht: "
                    f"{it['name']} ({code}) — Preis {px:,.2f} € ≤ SL {sl_level:,.2f} € "
                    f"(−{sl_pct:.1f}% vom High {high:,.2f} €)"
                )
                new_assets[code] = {
                    "price": px, "high": high, "low": low, "need_redeploy": True
                }

        if tp_pct > 0 and low == low and px == px:
            tp_level = low * (1 + tp_pct/100.0)
            if px >= tp_level:
                alerts.append(
                    "✅ Take-Profit erreicht: "
                    f"{it['name']} ({code}) — Preis {px:,.2f} € ≥ TP {tp_level:,.2f} € "
                    f"(+{tp_pct:.1f}% vom Low {low:,.2f} €)"
                )
                new_assets[code] = {
                    "price": px, "high": high, "low": low, "need_redeploy": True
                }

        if code not in new_assets:
            new_assets[code] = {"price": px, "high": high, "low": low}

    save_json(
        STATE_FILE,
        {
            "portfolio": {"total": total},
            "assets": new_assets
        }
    )
    return alerts

# ----------------------------- TA-Indikatoren / Scoring -----------------------

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

def score_stock(ticker: str) -> dict | None:
    close = yf_history(ticker, period="6mo", interval="1d")
    if close.empty or len(close) < 50:
        return None

    last = float(close.iloc[-1])
    rsi14 = float(rsi(close, 14).iloc[-1])

    macd_line, signal_line, hist = macd(close)
    macd_last = float(macd_line.iloc[-1])
    signal_last = float(signal_line.iloc[-1])
    hist_last = float(hist.iloc[-1])

    sma20 = sma(close, 20)
    sma50 = sma(close, 50)
    sma200 = sma(close, 200)

    sma20_l = float(sma20.iloc[-1]) if not np.isnan(sma20.iloc[-1]) else float("nan")
    sma50_l = float(sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else float("nan")
    sma200_l = float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else float("nan")

    ret_7d = (last / float(close.iloc[-7]) - 1) * 100 if len(close) >= 8 else 0.0
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

    if ret_7d > 2:
        score += 1; notes.append(f"7d {ret_7d:+.1f}%")
    if ret_30d > 4:
        score += 1; notes.append(f"30d {ret_30d:+.1f}%")

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

    last = float(close.iloc[-1])
    rsi14 = float(rsi(close, 14).iloc[-1])

    macd_line, signal_line, hist = macd(close)
    macd_last = float(macd_line.iloc[-1])
    signal_last = float(signal_line.iloc[-1])
    hist_last = float(hist.iloc[-1])

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

    if ret_7d > 3:
        score += 1; notes.append(f"7d {ret_7d:+.1f}%")
    if ret_30d > 6:
        score += 1; notes.append(f"30d {ret_30d:+.1f}%")

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
    """
    Vorschlag Positionsgröße für neue Ideen (Scanner),
    mit Puffer: kleines Depot soll nicht sofort All-in gehen.
    """
    pct_map = {"conservative": 0.02, "balanced": 0.03, "aggressive": 0.05}
    base_pct = pct_map.get(risk_level, 0.03)
    per_trade_eur = total_value * base_pct

    min_depot = 1500.0
    safety_buffer = max(total_value - min_depot, 0)
    per_trade_eur = min(per_trade_eur, safety_buffer * 0.2)

    return round(per_trade_eur, 2)

# ----------------------------- Watchlist Mgmt -----------------------------

def sync_watchlist_from_positions():
    """
    Erstellt/aktualisiert watchlist.json:
    - nimmt alle Ticker aus deinem Depot
    - ergänzt unser BASE_UNIVERSE_... + (optional) EXTRA_...
    - capped via MAX_STOCKS / MAX_CRYPTO
    """
    expand = os.getenv("EXPAND_UNIVERSE", "false").lower() in ("1","true","yes","y")

    pos = load_positions()
    pos_stocks = [
        _norm_code(s.get("ticker",""))
        for s in pos.get("stocks", [])
        if s.get("ticker")
    ]
    pos_crypto = [
        _norm_code(c.get("symbol",""))
        for c in pos.get("crypto", [])
        if c.get("symbol")
    ]

    pool_stocks = pos_stocks + BASE_UNIVERSE_STOCKS
    pool_crypto = pos_crypto + BASE_UNIVERSE_CRYPTO

    if expand:
        pool_stocks += EXTRA_UNIVERSE_STOCKS
        pool_crypto += EXTRA_UNIVERSE_CRYPTO

    stocks_final = _dedup_keep_order(pool_stocks)[:MAX_STOCKS]
    crypto_final = _dedup_keep_order(pool_crypto)[:MAX_CRYPTO]

    wl = load_json(WATCHLIST_FILE, {"stocks": [], "crypto": []})
    wl["stocks"] = _dedup_keep_order(stocks_final + wl.get("stocks", []))[:MAX_STOCKS]
    wl["crypto"] = _dedup_keep_order(crypto_final + wl.get("crypto", []))[:MAX_CRYPTO]

    save_json(WATCHLIST_FILE, wl)
    return wl

def load_watchlist():
    # immer zuerst synchronisieren
    return sync_watchlist_from_positions()

# ----------------------------- Scanner Job -------------------------------

def job_scanner():
    wl = load_watchlist()
    items, total = compute_portfolio()
    if total <= 0:
        return  # nix Scannen wenn wir gar keinen Wert haben

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
        notify("Chancenfinder", "Keine klaren Setups aktuell – Kapital lieber parken (Schutzmodus).")
        return

    size_eur = suggest_position_sizes(total, RISK_MODE)

    def fmt(entry):
        hint = f" • {entry['entry_hint']}" if entry.get("entry_hint") else ""
        return (
            f"{entry['code']}: Score {entry['score']} | "
            f"Preis ~{entry['price']:,.2f} € | "
            f"RSI {entry['rsi']:.0f}, MACDΔ {entry['macd']:+.3f}, "
            f"7d {entry['ret7']:+.1f}%, 30d {entry['ret30']:+.1f}%"
            f"{hint} — Vorschlag: ~{size_eur:,.2f} €"
        )

    lines = []
    if top_s:
        lines.append("📈 Aktien/ETFs – Top Setups:")
        lines += ["• " + fmt(e) for e in top_s]
    if top_c:
        lines.append("\n🪙 Krypto – Top Setups:")
        lines += ["• " + fmt(e) for e in top_c]

    notify("Chancenfinder – neue Reinvest-Ideen", "\n".join(lines))

# ----------------------------- Scheduler Jobs ---------------------------

def job_daily_summary():
    items, total = compute_portfolio()
    report = render_report(items, total)
    notify("Tagesbericht", report)

def job_intraday_monitor():
    """
    Läuft alle 15 Minuten werktags.
    - Außerhalb 06:10–22:00 Uhr DE: kein Push, aber State wird aktualisiert.
    - Innerhalb der Zeit: echte Alerts schicken.
    """
    t = datetime.now(TZ).time()
    in_window = (
        t >= datetime.strptime("06:10", "%H:%M").time() and
        t <= datetime.strptime("22:00", "%H:%M").time()
    )

    items, total = compute_portfolio()

    # Wenn Portfolio sehr klein ist, bitte keinen Alarmstress
    if total < MIN_ALERT_EUR:
        # trotzdem State updaten, damit spätere Prozentvergleiche Sinn machen
        state = load_json(STATE_FILE, {"portfolio": {"total": None}, "assets": {}})
        save_json(
            STATE_FILE,
            {
                "portfolio": {"total": total},
                "assets": state.get("assets", {})
            }
        )
        return

    # Alerts berechnen
    alerts = evaluate_alerts(items, total)

    # Nur während Handelsfenster pushen
    if in_window and alerts:
        notify("Intraday-Alarm", "\n".join(alerts))

# ----------------------------- Main ------------------------------------

def main():
    _dbg(f"Agent gestartet. Alpaca={'on' if ALPACA_ENABLED else 'off'}")

    # Watchlist initial sync
    sync_watchlist_from_positions()

    # initialen STATE_FILE anlegen, falls fehlt
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
                        "low": it["price"]
                    }
                    for it in items
                }
            }
        )
        _dbg("Initialer Zustand gespeichert.")

    # Scheduler konfigurieren
    sched = BlockingScheduler(timezone=TZ)

    # Tagesbericht 06:30 jeden Tag
    sched.add_job(job_daily_summary, CronTrigger(hour=6, minute=30))

    # Intraday-Überwachung alle 15 Min (Mo–Fr zwischen 06:10 und 22:00 berücksichtigt job selbst)
    sched.add_job(
        job_intraday_monitor,
        CronTrigger(day_of_week='mon-fri', hour='6-22', minute='*/15')
    )

    # Chancenfinder 4×/Tag (auch am Wochenende interessant)
    for hh, mm in [(7,15), (12,15), (16,15), (20,15)]:
        sched.add_job(
            job_scanner,
            CronTrigger(day_of_week='mon-sun', hour=hh, minute=mm)
        )

    _dbg("Scheduler aktiv. Warte auf Jobs …")

    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        _dbg("Agent gestoppt.")

if __name__ == "__main__":
    main()
