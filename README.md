# Portfolio Agent (Trade Republic – Analyse, Alerts & Chancenfinder)

## Start (lokal)
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Variablen befüllen
python monitor.py

## Render (24/7 empfohlen)
- New → Background Worker
- Build Command: pip install -r requirements.txt
- Start Command: python monitor.py
- Environment: .env-Variablen im Dashboard hinterlegen

## Dateien
- monitor.py        → Agent mit Tagesreport, Intraday-Alerts & Chancenfinder
- positions.json    → Dein Depot (editierbar; optional SL/TP pro Position)
- watchlist.json    → Wird beim Start automatisch aus Depot + Universen gepflegt
- .env              → Telegram/E-Mail & Scan-Parameter (MAX_STOCKS/MAX_CRYPTO)
- requirements.txt  → Python-Abhängigkeiten

## Zeiten
- 06:30  Tagesbericht
- 06:10–22:00 Intraday-Überwachung alle 15 Minuten (Mo–Fr)
- 07:15 / 12:15 / 16:15 / 20:15  Chancenfinder (täglich)

## Hinweise
- Kein Auto-Trading. Der Agent analysiert & sendet Signale.
- Mindestdepot 1.500 € wird respektiert (Positionsgrößen berücksichtigen Puffer).
