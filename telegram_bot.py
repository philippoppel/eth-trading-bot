#!/usr/bin/env python3
"""
Interaktiver Telegram Bot für Trading Status.
Reagiert auf Befehle wie /status, /performance, /price
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Projekt-Root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from src.data.binance_client import BinanceClient

# Status-Datei für Paper Trader
STATUS_FILE = PROJECT_ROOT / 'data' / 'trading_status.json'


def load_trading_status() -> dict:
    """Lädt aktuellen Trading-Status."""
    if STATUS_FILE.exists():
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    return None


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start-Befehl."""
    await update.message.reply_text(
        "🤖 <b>ETH Trading Bot</b>\n\n"
        "Verfügbare Befehle:\n"
        "/status - Aktueller Bot-Status\n"
        "/price - Aktueller ETH Preis\n"
        "/performance - Performance-Übersicht\n"
        "/trades - Letzte Trades\n"
        "/logs - Letzte Entscheidungen\n"
        "/help - Diese Hilfe",
        parse_mode='HTML'
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Hilfe-Befehl."""
    await cmd_start(update, context)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Status-Befehl."""
    status = load_trading_status()

    if status:
        position_text = "Keine Position" if status.get('position', 0) == 0 else f"{status['position']:.4f} ETH"
        pnl = status.get('pnl', 0)
        pnl_pct = status.get('pnl_pct', 0)
        pnl_emoji = "📈" if pnl >= 0 else "📉"

        msg = f"""
📊 <b>Trading Bot Status</b>

💰 Balance: ${status.get('balance', 10000):,.2f}
📦 Position: {position_text}
💵 Portfolio: ${status.get('portfolio_value', 10000):,.2f}
{pnl_emoji} PnL: ${pnl:,.2f} ({pnl_pct:+.2f}%)
📈 Trades: {status.get('num_trades', 0)}

🕐 Update: {status.get('last_update', 'N/A')}
"""
    else:
        # Kein Status vorhanden - zeige Basis-Info
        try:
            client = BinanceClient(testnet=True)
            price = client.get_current_price('ETH/USDT')
            msg = f"""
📊 <b>Trading Bot Status</b>

⚠️ Bot läuft nicht oder keine Daten

💵 Aktueller ETH Preis: ${price:,.2f}

Starte den Bot mit:
<code>python run_bot.py --model lstm</code>
"""
        except:
            msg = "📊 <b>Status</b>\n\n⚠️ Keine Daten verfügbar. Bot nicht gestartet?"

    await update.message.reply_text(msg, parse_mode='HTML')


async def cmd_price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Preis-Befehl."""
    try:
        client = BinanceClient(testnet=True)
        price = client.get_current_price('ETH/USDT')

        msg = f"""
💰 <b>ETH/USDT Preis</b>

${price:,.2f}

🕐 {datetime.now().strftime('%H:%M:%S')}
"""
    except Exception as e:
        msg = f"❌ Fehler beim Abrufen: {e}"

    await update.message.reply_text(msg, parse_mode='HTML')


async def cmd_performance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Performance-Befehl."""
    status = load_trading_status()

    if status and 'metrics' in status:
        metrics = status['metrics']
        msg = f"""
📈 <b>Performance Übersicht</b>

💰 Start: $10,000.00
💵 Aktuell: ${status.get('portfolio_value', 10000):,.2f}

📊 Total Return: {metrics.get('total_return', 0)*100:+.2f}%
📉 Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%
🎯 Win Rate: {metrics.get('win_rate', 0)*100:.1f}%
📈 Trades: {metrics.get('num_trades', 0)}
"""
    elif status:
        pnl = status.get('pnl', 0)
        pnl_pct = status.get('pnl_pct', 0)
        msg = f"""
📈 <b>Performance Übersicht</b>

💰 Start: $10,000.00
💵 Aktuell: ${status.get('portfolio_value', 10000):,.2f}

📊 PnL: ${pnl:,.2f} ({pnl_pct:+.2f}%)
📈 Trades: {status.get('num_trades', 0)}
"""
    else:
        msg = "📈 <b>Performance</b>\n\n⚠️ Keine Daten. Bot läuft nicht."

    await update.message.reply_text(msg, parse_mode='HTML')


async def cmd_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Trades-Befehl."""
    status = load_trading_status()

    if status and 'recent_trades' in status and status['recent_trades']:
        trades = status['recent_trades'][-5:]  # Letzte 5 Trades

        msg = "📋 <b>Letzte Trades</b>\n\n"
        for trade in reversed(trades):
            emoji = "🟢" if trade['type'] == 'buy' else "🔴"
            pnl_text = ""
            if 'pnl' in trade:
                pnl_text = f" | PnL: ${trade['pnl']:+.2f}"
            msg += f"{emoji} {trade['type'].upper()} @ ${trade['price']:,.2f}{pnl_text}\n"
    else:
        msg = "📋 <b>Trades</b>\n\n⚠️ Keine Trades bisher."

    await update.message.reply_text(msg, parse_mode='HTML')


async def cmd_logs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Zeigt letzte Log-Einträge."""
    try:
        log_file = PROJECT_ROOT / 'logs' / 'trading.log'

        if not log_file.exists():
            await update.message.reply_text("📜 Log-Datei nicht gefunden\n\nStarte erst den Trading Bot.")
            return

        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        msg = "📜 <b>Letzte Entscheidungen</b>\n\n"

        # Finde Prediction-Zeilen
        predictions = [l for l in lines if 'LSTM Prediction' in l][-5:]

        if not predictions:
            await update.message.reply_text("📜 Keine Entscheidungen gefunden\n\nWarte auf erste Prediction.")
            return

        for line in predictions:
            try:
                # Zeit extrahieren
                time_part = line.split(' ')[1][:5] if ' ' in line else "??:??"

                # Prediction extrahieren
                if 'Prediction: 0' in line:
                    signal = "🔴 DOWN"
                elif 'Prediction: 1' in line:
                    signal = "⏸️ HOLD"
                elif 'Prediction: 2' in line:
                    signal = "🟢 UP"
                else:
                    signal = "?"

                # Confidence extrahieren
                conf_start = line.find('Confidence: ')
                if conf_start != -1:
                    conf = line[conf_start + 12:conf_start + 16]
                else:
                    conf = "?"

                msg += f"<code>{time_part}</code> {signal} ({conf})\n"
            except Exception:
                continue

        # Aktueller Preis aus Status-Datei
        status = load_trading_status()
        if status and 'current_price' in status:
            msg += f"\n💰 Aktuell: <b>${status['current_price']:,.2f}</b>"

        await update.message.reply_text(msg, parse_mode='HTML')

    except Exception as e:
        await update.message.reply_text(f"❌ Fehler: {str(e)[:100]}")


def main():
    """Startet den Bot."""
    token = os.getenv('TELEGRAM_BOT_TOKEN')

    if not token:
        print("Fehler: TELEGRAM_BOT_TOKEN nicht gesetzt!")
        return

    print("🤖 Telegram Bot startet...")
    print("Befehle: /status, /price, /performance, /trades")
    print("Drücke Ctrl+C zum Beenden")

    # Application erstellen
    app = Application.builder().token(token).build()

    # Handler hinzufügen
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("price", cmd_price))
    app.add_handler(CommandHandler("performance", cmd_performance))
    app.add_handler(CommandHandler("trades", cmd_trades))
    app.add_handler(CommandHandler("logs", cmd_logs))

    # Bot starten
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
