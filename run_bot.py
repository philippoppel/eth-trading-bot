#!/usr/bin/env python3
"""
Haupt-Script zum Starten des Trading Bots.
"""

import sys
import argparse
from pathlib import Path
from loguru import logger

# Projekt-Root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.trading.paper_trader import PaperTrader


def setup_logging():
    """Konfiguriert Logging."""
    log_path = PROJECT_ROOT / 'logs' / 'trading.log'
    log_path.parent.mkdir(exist_ok=True)

    logger.add(
        str(log_path),
        rotation="1 day",
        retention="7 days",
        level="INFO"
    )


def main():
    """Hauptfunktion."""
    parser = argparse.ArgumentParser(description='ETH Trading Bot')

    parser.add_argument(
        '--model',
        type=str,
        choices=['lstm', 'rl'],
        default='lstm',
        help='Model Typ (lstm oder rl)'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Pfad zum trainierten Modell'
    )

    parser.add_argument(
        '--once',
        action='store_true',
        help='Nur einen Zyklus ausführen'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Zeigt nur Status an'
    )

    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Führt Backtest durch'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Backtest Tage'
    )

    args = parser.parse_args()

    # Logging einrichten
    setup_logging()

    # Backtest Modus
    if args.backtest:
        run_backtest(args)
        return

    # Trader initialisieren
    trader = PaperTrader(
        config_path=str(PROJECT_ROOT / 'config.yaml'),
        model_type=args.model
    )

    # Model laden
    if args.model_path:
        model_path = args.model_path
    else:
        if args.model == 'lstm':
            model_path = str(PROJECT_ROOT / 'models' / 'lstm_best.pt')
        else:
            model_path = str(PROJECT_ROOT / 'models' / 'ppo_trading.zip')

    # Prüfen ob Modell existiert
    if Path(model_path).exists():
        if args.model == 'lstm':
            trader.load_lstm_model(model_path)
        else:
            trader.load_rl_model(model_path)
    else:
        logger.warning(f"Modell nicht gefunden: {model_path}")
        logger.warning("Starte ohne Modell (nur Status)")
        args.status = True

    # Status Modus
    if args.status:
        status = trader.get_status()
        print("\n" + "=" * 50)
        print("TRADING BOT STATUS")
        print("=" * 50)
        print(f"Symbol: {status['symbol']}")
        print(f"Timeframe: {status['timeframe']}")
        print(f"Balance: ${status['balance']:.2f}")
        print(f"Position: {status['position']:.4f}")
        print(f"Current Price: ${status['current_price']:.2f}")
        print(f"Portfolio Value: ${status['portfolio_value']:.2f}")
        print(f"PnL: ${status['pnl']:.2f} ({status['pnl_pct']:+.2f}%)")
        print(f"Trades: {status['num_trades']}")
        print("=" * 50)
        return

    # Einmal-Modus
    if args.once:
        logger.info("Führe einen Trading-Zyklus aus...")
        trader.run_once()
        return

    # Kontinuierlicher Modus
    try:
        logger.info("Starte kontinuierlichen Trading-Modus...")
        trader.run()
    except KeyboardInterrupt:
        logger.info("Trading Bot durch Benutzer gestoppt")
        trader.stop()


def run_backtest(args):
    """Führt Backtest durch."""
    from src.data.binance_client import BinanceClient
    from src.data.features import FeatureEngineer
    from src.trading.backtest import Backtester
    import yaml

    logger.info("=" * 60)
    logger.info("BACKTEST MODUS")
    logger.info("=" * 60)

    # Config laden
    with open(PROJECT_ROOT / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Daten laden
    client = BinanceClient(testnet=True)
    df = client.get_historical_data(
        config['trading']['symbol'],
        config['trading']['timeframe'],
        days=args.days
    )

    # Features
    engineer = FeatureEngineer()
    df = engineer.process(df, normalize=True)

    # Signal-Funktion basierend auf Modell
    if args.model == 'lstm' and Path(PROJECT_ROOT / 'models' / 'lstm_best.pt').exists():
        # LSTM-basiertes Signal
        from src.models.lstm import LSTMModel, LSTMTrainer

        model = LSTMModel(
            input_size=len([c for c in df.columns if c.endswith('_norm')]),
            hidden_size=config['model']['lstm']['hidden_size'],
            num_layers=config['model']['lstm']['num_layers']
        )
        trainer = LSTMTrainer(model)
        trainer.load_model(str(PROJECT_ROOT / 'models' / 'lstm_best.pt'))

        def lstm_signal(df, i):
            if i < 60:
                return 0
            feature_cols = [c for c in df.columns if c.endswith('_norm')]
            features = df[feature_cols].iloc[i-60:i].values
            X = features.reshape(1, 60, -1)
            pred, probs = trainer.predict(X)
            if probs[0].max() < 0.4:  # Schwelle über Zufall (33%)
                return 0
            if pred[0] == 2:
                return 1
            elif pred[0] == 0:
                return -1
            return 0

        signal_func = lstm_signal
        logger.info("Verwende LSTM Modell für Signale")

    else:
        # Einfaches RSI-Signal
        def rsi_signal(df, i):
            if i < 14:
                return 0
            rsi = df.iloc[i].get('rsi', 50)
            if rsi < 30:
                return 1
            elif rsi > 70:
                return -1
            return 0

        signal_func = rsi_signal
        logger.info("Verwende RSI-basierte Signale")

    # Backtest
    backtester = Backtester(
        initial_balance=config['trading']['initial_balance'],
        fee_rate=config['backtest']['fee_rate'],
        slippage_rate=config['backtest']['slippage_rate'],
        stop_loss_pct=config['risk']['stop_loss_pct'],
        take_profit_pct=config['risk']['take_profit_pct'],
        max_position_pct=config['risk']['max_position_pct']
    )

    result = backtester.run(df, signal_func)
    backtester.print_results(result)


if __name__ == '__main__':
    main()
