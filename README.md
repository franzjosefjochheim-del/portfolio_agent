# Portfolio Agent (Trade Republic – Analyse & Alerts)

## Schnellstart lokal
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Variablen eintragen
python monitor.py     # startet Scheduler (Daily + Intraday)

## Render (empfohlen)
1) Repo auf GitHub pushen.
2) Render → New → Web Service (oder Background Worker).
3) Build Command: `pip install -r requirements.txt`
4) Start Command: `python monitor.py`
5) Environment: `.env`-Variablen im Dashboard hinterlegen.
6) Service dauerhaft laufen lassen (Background Worker).

## Replit
- Neues Repl (Python), Dateien einfügen.
- Secrets aus `.env.example` als Replit Secrets anlegen.
- `pip install -r requirements.txt`
- Run: `python monitor.py`

## Regeln (Stop/Take)
In `positions.json` optional pro Position:
- `stop_loss_pct`: Trailing-SL relativ zum beobachteten High
- `take_profit_pct`: Take-Profit relativ zum beobachteten Low

## Benachrichtigungen
- Telegram via Bot-Token & Chat-ID
- E-Mail via SMTP (optional)

## Hinweise
- Das Skript führt **keine Trades** aus. Es analysiert, alarmiert und erstellt Reports.
- Handelsentscheidungen triffst du selbst in Trade Republic.
