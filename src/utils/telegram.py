"""
Telegram Bot für Trading Alerts.
"""

import os
import asyncio
from datetime import datetime
from loguru import logger

try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot nicht installiert. Telegram Alerts deaktiviert.")


class TelegramBot:
    """Telegram Bot für Trading Alerts."""

    def __init__(self):
        """Initialisiert Telegram Bot."""
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.enabled = bool(self.bot_token and self.chat_id and TELEGRAM_AVAILABLE)

        if self.enabled:
            self.bot = Bot(token=self.bot_token)
            logger.info("Telegram Bot initialisiert")
        else:
            self.bot = None
            if not self.bot_token:
                logger.info("Telegram: Kein Bot Token konfiguriert")
            if not self.chat_id:
                logger.info("Telegram: Keine Chat ID konfiguriert")

    def _send_sync(self, message: str):
        """Sendet Nachricht synchron."""
        if not self.enabled:
            return

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            ))
            loop.close()
        except TelegramError as e:
            logger.error(f"Telegram Fehler: {e}")
        except Exception as e:
            logger.error(f"Telegram Fehler: {e}")

    def send_message(self, message: str):
        """Sendet einfache Nachricht."""
        self._send_sync(message)

    def send_trade_alert(
        self,
        action: str,
        symbol: str,
        price: float,
        amount: float,
        pnl: float = None
    ):
        """
        Sendet Trade Alert.

        Args:
            action: 'BUY' oder 'SELL'
            symbol: Trading Pair
            price: Ausführungspreis
            amount: Menge
            pnl: Profit/Loss (nur bei SELL)
        """
        emoji = "🟢" if action == "BUY" else "🔴"
        timestamp = datetime.now().strftime("%H:%M:%S")

        message = f"""
{emoji} <b>{action} {symbol}</b>

📊 Preis: ${price:,.2f}
📦 Menge: {amount:.4f}
🕐 Zeit: {timestamp}
"""

        if pnl is not None:
            pnl_emoji = "✅" if pnl > 0 else "❌"
            message += f"\n{pnl_emoji} PnL: ${pnl:,.2f}"

        self._send_sync(message)

    def send_daily_report(
        self,
        symbol: str,
        portfolio_value: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        num_trades: int,
        win_rate: float
    ):
        """Sendet täglichen Report."""
        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
        timestamp = datetime.now().strftime("%Y-%m-%d")

        message = f"""
📊 <b>Daily Report - {timestamp}</b>

💰 Portfolio: ${portfolio_value:,.2f}
{pnl_emoji} Tages-PnL: ${daily_pnl:,.2f} ({daily_pnl_pct:+.2f}%)
📈 Trades: {num_trades}
🎯 Win Rate: {win_rate*100:.1f}%

Symbol: {symbol}
"""

        self._send_sync(message)

    def send_risk_alert(self, alert_type: str, details: str):
        """
        Sendet Risiko-Warnung.

        Args:
            alert_type: 'drawdown', 'daily_loss', 'kill_switch'
            details: Zusätzliche Details
        """
        emoji_map = {
            'drawdown': '⚠️',
            'daily_loss': '🚨',
            'kill_switch': '🛑'
        }
        emoji = emoji_map.get(alert_type, '⚠️')

        message = f"""
{emoji} <b>RISIKO WARNUNG</b>

Typ: {alert_type.upper()}
{details}

🕐 Zeit: {datetime.now().strftime("%H:%M:%S")}
"""

        self._send_sync(message)

    def send_error_alert(self, error: str):
        """Sendet Fehler-Alert."""
        message = f"""
❌ <b>FEHLER</b>

{error}

🕐 Zeit: {datetime.now().strftime("%H:%M:%S")}
"""

        self._send_sync(message)

    def send_startup_message(self, symbol: str, balance: float):
        """Sendet Startup-Nachricht."""
        message = f"""
🚀 <b>Trading Bot Gestartet</b>

📊 Symbol: {symbol}
💰 Balance: ${balance:,.2f}
🕐 Zeit: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

        self._send_sync(message)


if __name__ == "__main__":
    # Test (benötigt konfigurierte Umgebungsvariablen)
    bot = TelegramBot()

    if bot.enabled:
        bot.send_message("Test Nachricht vom Trading Bot!")
        bot.send_trade_alert('BUY', 'ETH/USDT', 2000.50, 0.5)
    else:
        print("Telegram nicht konfiguriert. Setze TELEGRAM_BOT_TOKEN und TELEGRAM_CHAT_ID in .env")
