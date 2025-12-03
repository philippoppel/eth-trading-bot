"""
Paper Trading Bot für ETH/USDT.
Führt Live-Trading auf Binance Testnet aus.
"""

import time
import json
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.binance_client import BinanceClient
from src.data.features import FeatureEngineer
from src.trading.risk import RiskManager


class PaperTrader:
    """Paper Trading Bot."""

    def __init__(
        self,
        config_path: str = 'config.yaml',
        model_type: str = 'lstm'
    ):
        """
        Args:
            config_path: Pfad zur Konfiguration
            model_type: 'lstm' oder 'rl'
        """
        # Config laden
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_type = model_type
        self.symbol = self.config['trading']['symbol']
        self.timeframe = self.config['trading']['timeframe']
        self.initial_balance = self.config['trading']['initial_balance']

        # Komponenten initialisieren
        self.client = BinanceClient(testnet=True)
        self.feature_engineer = FeatureEngineer()
        self.risk_manager = RiskManager(
            initial_balance=self.initial_balance,
            max_position_pct=self.config['risk']['max_position_pct'],
            stop_loss_pct=self.config['risk']['stop_loss_pct'],
            take_profit_pct=self.config['risk']['take_profit_pct'],
            daily_loss_limit_pct=self.config['risk']['daily_loss_limit_pct'],
            max_drawdown_pct=self.config['risk']['max_drawdown_pct']
        )

        # Model laden
        self.model = None
        self.model_loaded = False

        # State
        self.position = 0.0
        self.position_price = 0.0
        self.balance = self.initial_balance
        self.trades = []
        self.running = False

        # Telegram (optional)
        self.telegram = None
        if self.config['telegram']['enabled']:
            from src.utils.telegram import TelegramBot
            self.telegram = TelegramBot()

        logger.info(f"Paper Trader initialisiert: {self.symbol} @ {self.timeframe}")

    def load_lstm_model(self, model_path: str):
        """Lädt LSTM Modell."""
        import torch
        from src.models.lstm import LSTMModel, LSTMTrainer

        # Modell-Konfiguration
        checkpoint = torch.load(model_path, map_location='cpu')
        input_size = checkpoint['model_state_dict']['lstm.weight_ih_l0'].shape[1]

        model = LSTMModel(
            input_size=input_size,
            hidden_size=self.config['model']['lstm']['hidden_size'],
            num_layers=self.config['model']['lstm']['num_layers'],
            dropout=self.config['model']['lstm']['dropout']
        )

        self.model = LSTMTrainer(model)
        self.model.load_model(model_path)
        self.model_loaded = True
        logger.info(f"LSTM Modell geladen: {model_path}")

    def load_rl_model(self, model_path: str):
        """Lädt RL Agent."""
        from src.models.rl_agent import load_ppo_agent

        self.model = load_ppo_agent(model_path)
        self.model_loaded = True
        logger.info(f"RL Agent geladen: {model_path}")

    def get_signal_lstm(self, df: pd.DataFrame) -> int:
        """
        Generiert Signal mit LSTM Modell.

        Returns:
            Signal: 1=Buy, -1=Sell, 0=Hold
        """
        if not self.model_loaded:
            logger.warning("LSTM Modell nicht geladen!")
            return 0

        # Features vorbereiten
        feature_cols = [c for c in df.columns if c.endswith('_norm')]
        features = df[feature_cols].values

        # Sequenz erstellen (letzte 60 Zeitschritte)
        seq_length = self.config['data']['sequence_length']
        if len(features) < seq_length:
            return 0

        X = features[-seq_length:].reshape(1, seq_length, -1)

        # Prediction
        pred_class, probs = self.model.predict(X)
        pred = pred_class[0]
        confidence = probs[0].max()

        logger.debug(f"LSTM Prediction: {pred} (Confidence: {confidence:.2f})")

        # Signal generieren (Schwelle über Zufall 33%)
        if confidence < 0.4:
            return 0

        if pred == 2:  # Up
            return 1
        elif pred == 0:  # Down
            return -1

        return 0

    def get_signal_rl(self, df: pd.DataFrame) -> int:
        """
        Generiert Signal mit RL Agent.

        Returns:
            Signal: 1=Buy, -1=Sell, 0=Hold
        """
        if not self.model_loaded:
            logger.warning("RL Agent nicht geladen!")
            return 0

        # State erstellen (vereinfacht)
        feature_cols = [c for c in df.columns if c.endswith('_norm')]
        obs_window = self.config['data']['sequence_length']

        features = df[feature_cols].tail(obs_window).values.flatten()

        # Position Info hinzufügen
        position_normalized = self.position * df.iloc[-1]['close'] / self.balance if self.balance > 0 else 0
        unrealized_pnl = 0
        if self.position > 0:
            unrealized_pnl = (df.iloc[-1]['close'] - self.position_price) / self.position_price

        state = np.concatenate([features, [position_normalized, unrealized_pnl]]).astype(np.float32)

        # Prediction
        action, _ = self.model.predict(state, deterministic=True)

        # Action zu Signal konvertieren
        if action == 1:  # Buy
            return 1
        elif action == 2:  # Sell
            return -1

        return 0

    def execute_trade(self, signal: int, current_price: float):
        """Führt Trade aus."""
        if signal == 0:
            return

        risk_state = self.risk_manager.check_risk()

        if not risk_state.can_trade:
            logger.warning(f"Trading blockiert: {risk_state.reason}")
            return

        timestamp = datetime.now()

        if signal == 1 and self.position == 0:  # Buy
            # Position Size berechnen
            position_size = self.risk_manager.calculate_position_size(current_price)

            if position_size <= 0:
                return

            # Order platzieren
            try:
                order = self.client.place_market_order(
                    self.symbol.replace('/', ''),
                    'buy',
                    position_size
                )

                self.position = position_size
                self.position_price = current_price

                # Balance aktualisieren (vereinfacht)
                cost = position_size * current_price * 1.001  # inkl. Fees
                self.balance -= cost

                trade = {
                    'time': timestamp,
                    'type': 'buy',
                    'price': current_price,
                    'size': position_size,
                    'order_id': order.get('id')
                }
                self.trades.append(trade)

                logger.info(f"BUY: {position_size:.4f} ETH @ ${current_price:.2f}")

                # Telegram Alert
                if self.telegram:
                    self.telegram.send_trade_alert('BUY', self.symbol, current_price, position_size)

            except Exception as e:
                logger.error(f"Buy Order fehlgeschlagen: {e}")

        elif signal == -1 and self.position > 0:  # Sell
            try:
                order = self.client.place_market_order(
                    self.symbol.replace('/', ''),
                    'sell',
                    self.position
                )

                # PnL berechnen
                sell_value = self.position * current_price * 0.999  # inkl. Fees
                pnl = sell_value - (self.position * self.position_price)
                pnl_pct = (current_price / self.position_price - 1) * 100

                self.balance += sell_value

                trade = {
                    'time': timestamp,
                    'type': 'sell',
                    'price': current_price,
                    'size': self.position,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'order_id': order.get('id')
                }
                self.trades.append(trade)

                # Risk Manager updaten
                self.risk_manager.record_trade(pnl)
                self.risk_manager.update_balance(self.balance)

                logger.info(f"SELL: {self.position:.4f} ETH @ ${current_price:.2f} | PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")

                # Telegram Alert
                if self.telegram:
                    self.telegram.send_trade_alert('SELL', self.symbol, current_price, self.position, pnl)

                self.position = 0.0
                self.position_price = 0.0

            except Exception as e:
                logger.error(f"Sell Order fehlgeschlagen: {e}")

    def check_stop_loss_take_profit(self, current_price: float):
        """Prüft und führt Stop Loss / Take Profit aus."""
        if self.position == 0:
            return

        stop_loss = self.risk_manager.calculate_stop_loss(self.position_price)
        take_profit = self.risk_manager.calculate_take_profit(self.position_price)

        if current_price <= stop_loss:
            logger.warning(f"STOP LOSS triggered @ ${current_price:.2f}")
            self.execute_trade(-1, current_price)
        elif current_price >= take_profit:
            logger.info(f"TAKE PROFIT triggered @ ${current_price:.2f}")
            self.execute_trade(-1, current_price)

    def run_once(self):
        """Führt einen Trading-Zyklus aus."""
        try:
            # Aktuelle Daten laden
            df = self.client.get_historical_data(
                self.symbol,
                self.timeframe,
                days=7  # Letzte Woche für Features
            )

            # Features berechnen
            df = self.feature_engineer.process(df)

            if len(df) < self.config['data']['sequence_length']:
                logger.warning("Nicht genug Daten")
                return

            current_price = df.iloc[-1]['close']

            # Stop Loss / Take Profit prüfen
            self.check_stop_loss_take_profit(current_price)

            # Signal generieren
            if self.model_type == 'lstm':
                signal = self.get_signal_lstm(df)
            else:
                signal = self.get_signal_rl(df)

            # Trade ausführen
            if signal != 0:
                self.execute_trade(signal, current_price)

            # Status loggen
            portfolio_value = self.balance + self.position * current_price
            logger.info(
                f"[{datetime.now().strftime('%H:%M')}] "
                f"ETH: ${current_price:.2f} | "
                f"Position: {self.position:.4f} | "
                f"Portfolio: ${portfolio_value:.2f}"
            )

            # Status für Telegram speichern
            self.save_status(current_price)

        except Exception as e:
            logger.error(f"Trading Zyklus Fehler: {e}")
            if self.telegram:
                self.telegram.send_error_alert(str(e))

    def save_status(self, current_price: float):
        """Speichert Status für Telegram Bot."""
        portfolio_value = self.balance + self.position * current_price
        pnl = portfolio_value - self.initial_balance
        pnl_pct = (pnl / self.initial_balance) * 100

        status = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'balance': self.balance,
            'position': self.position,
            'entry_price': getattr(self, 'entry_price', 0),
            'current_price': current_price,
            'portfolio_value': portfolio_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'num_trades': len(self.trades),
            'recent_trades': self.trades[-10:] if self.trades else [],
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        status_file = Path(__file__).parent.parent.parent / 'data' / 'trading_status.json'
        status_file.parent.mkdir(exist_ok=True)
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)

    def run(self, interval_seconds: int = None):
        """
        Startet kontinuierlichen Trading Loop.

        Args:
            interval_seconds: Intervall zwischen Checks (default: Timeframe-basiert)
        """
        if interval_seconds is None:
            # Timeframe zu Sekunden
            tf_map = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600, '4h': 14400}
            interval_seconds = tf_map.get(self.timeframe, 900)

        self.running = True
        logger.info(f"Starte Trading Loop (Intervall: {interval_seconds}s)")

        if self.telegram:
            self.telegram.send_message(f"Trading Bot gestartet: {self.symbol}")

        while self.running:
            self.run_once()

            # Warte bis zum nächsten Intervall
            time.sleep(interval_seconds)

    def stop(self):
        """Stoppt Trading Loop."""
        self.running = False
        logger.info("Trading Loop gestoppt")

        if self.telegram:
            self.telegram.send_message("Trading Bot gestoppt")

    def get_status(self) -> dict:
        """Gibt aktuellen Status zurück."""
        current_price = self.client.get_current_price(self.symbol)
        portfolio_value = self.balance + self.position * current_price

        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'balance': self.balance,
            'position': self.position,
            'position_price': self.position_price,
            'current_price': current_price,
            'portfolio_value': portfolio_value,
            'pnl': portfolio_value - self.initial_balance,
            'pnl_pct': (portfolio_value / self.initial_balance - 1) * 100,
            'num_trades': len(self.trades),
            'risk': self.risk_manager.get_status()
        }


if __name__ == "__main__":
    # Test
    trader = PaperTrader(model_type='lstm')

    # Ohne Modell: nur Status zeigen
    status = trader.get_status()
    print(f"Status: {status}")

    # Ein Zyklus ohne Modell (Signal = 0)
    trader.run_once()
