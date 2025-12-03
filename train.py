#!/usr/bin/env python3
"""
Training Script für ETH Trading Bot.
Trainiert LSTM und RL Modelle.
"""

import sys
import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger

# Projekt-Root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.binance_client import BinanceClient
from src.data.features import FeatureEngineer
from src.models.lstm import LSTMModel, LSTMTrainer, create_sequences, create_labels
from src.models.rl_agent import TradingEnv, train_ppo


def load_config():
    """Lädt Konfiguration."""
    with open(PROJECT_ROOT / 'config.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_data(config):
    """Lädt und bereitet Daten vor."""
    logger.info("=" * 60)
    logger.info("DATEN LADEN")
    logger.info("=" * 60)

    client = BinanceClient(testnet=True)

    # Historische Daten laden
    df = client.get_historical_data(
        symbol=config['trading']['symbol'],
        timeframe=config['trading']['timeframe'],
        days=config['data']['lookback_days']
    )

    # Daten speichern
    client.save_data(df, f"eth_usdt_{config['trading']['timeframe']}")

    logger.info(f"Geladen: {len(df)} Kerzen")
    logger.info(f"Zeitraum: {df.index[0]} bis {df.index[-1]}")

    return df


def prepare_features(df, config):
    """Berechnet Features."""
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 60)

    engineer = FeatureEngineer()
    df = engineer.process(df)

    logger.info(f"Features: {len(engineer.feature_columns)}")
    logger.info(f"Normalisierte Features: {len([c for c in df.columns if c.endswith('_norm')])}")
    logger.info(f"Finale Datenlänge: {len(df)}")

    return df, engineer


def train_lstm(df, config):
    """Trainiert LSTM Modell."""
    logger.info("=" * 60)
    logger.info("LSTM TRAINING")
    logger.info("=" * 60)

    # Feature-Spalten
    feature_cols = [c for c in df.columns if c.endswith('_norm')]
    features = df[feature_cols].values

    logger.info(f"Feature Dimension: {features.shape}")

    # Labels erstellen
    labels = create_labels(df['close'].values, threshold=0.005, lookahead=4)

    logger.info(f"Label Verteilung:")
    for cls, name in [(0, 'Down'), (1, 'Side'), (2, 'Up')]:
        count = (labels == cls).sum()
        logger.info(f"  {name}: {count} ({count/len(labels)*100:.1f}%)")

    # Sequenzen erstellen
    seq_length = config['data']['sequence_length']
    X, y = create_sequences(features[:-1], labels, seq_length)

    logger.info(f"Sequenzen: {X.shape}")

    # Train/Val/Test Split
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Model erstellen
    model = LSTMModel(
        input_size=X.shape[2],
        hidden_size=config['model']['lstm']['hidden_size'],
        num_layers=config['model']['lstm']['num_layers'],
        dropout=config['model']['lstm']['dropout']
    )

    trainer = LSTMTrainer(
        model,
        learning_rate=config['model']['lstm']['learning_rate']
    )

    # Training
    save_path = PROJECT_ROOT / 'models' / 'lstm_best.pt'

    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=config['model']['lstm']['epochs'],
        batch_size=config['model']['lstm']['batch_size'],
        early_stopping_patience=config['model']['lstm']['early_stopping_patience'],
        save_path=str(save_path)
    )

    # Evaluation
    result = trainer.evaluate(X_test, y_test)

    logger.info("=" * 40)
    logger.info("LSTM ERGEBNISSE")
    logger.info("=" * 40)
    logger.info(f"Test Accuracy: {result['accuracy']*100:.2f}%")
    logger.info("Class Accuracy:")
    for cls, acc in result['class_accuracy'].items():
        logger.info(f"  Class {cls}: {acc*100:.2f}%")

    return trainer, history, result


def train_rl(df, config):
    """Trainiert RL Agent."""
    logger.info("=" * 60)
    logger.info("RL AGENT TRAINING")
    logger.info("=" * 60)

    # Feature-Spalten
    feature_cols = [c for c in df.columns if c.endswith('_norm')]

    # Train/Test Split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)

    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Environments
    train_env = TradingEnv(
        df=train_df,
        feature_columns=feature_cols,
        initial_balance=config['trading']['initial_balance'],
        fee_rate=config['backtest']['fee_rate'],
        slippage_rate=config['backtest']['slippage_rate'],
        observation_window=config['data']['sequence_length']
    )

    test_env = TradingEnv(
        df=test_df,
        feature_columns=feature_cols,
        initial_balance=config['trading']['initial_balance'],
        fee_rate=config['backtest']['fee_rate'],
        slippage_rate=config['backtest']['slippage_rate'],
        observation_window=config['data']['sequence_length']
    )

    # Training
    save_path = str(PROJECT_ROOT / 'models' / 'ppo_trading')

    agent = train_ppo(
        train_env,
        eval_env=test_env,
        total_timesteps=config['model']['rl']['total_timesteps'],
        save_path=save_path
    )

    # Evaluation
    logger.info("Evaluiere auf Testdaten...")

    obs, _ = test_env.reset()
    done = False

    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated

    metrics = test_env.get_metrics()

    logger.info("=" * 40)
    logger.info("RL AGENT ERGEBNISSE")
    logger.info("=" * 40)
    logger.info(f"Total Return: {metrics['total_return']*100:.2f}%")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    logger.info(f"Win Rate: {metrics['win_rate']*100:.2f}%")
    logger.info(f"Trades: {metrics['num_trades']}")
    logger.info(f"Final Value: ${metrics['final_value']:.2f}")

    return agent, metrics


def main():
    """Hauptfunktion."""
    logger.info("=" * 60)
    logger.info("ETH TRADING BOT - MODEL TRAINING")
    logger.info(f"Gestartet: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Verzeichnisse erstellen
    (PROJECT_ROOT / 'models').mkdir(exist_ok=True)
    (PROJECT_ROOT / 'data').mkdir(exist_ok=True)
    (PROJECT_ROOT / 'logs').mkdir(exist_ok=True)

    # Config laden
    config = load_config()

    # Daten laden
    df = load_data(config)

    # Features berechnen
    df, engineer = prepare_features(df, config)

    # LSTM Training
    lstm_trainer, lstm_history, lstm_result = train_lstm(df, config)

    # RL Training
    rl_agent, rl_metrics = train_rl(df, config)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING ZUSAMMENFASSUNG")
    logger.info("=" * 60)

    logger.info("\nLSTM Modell:")
    logger.info(f"  Accuracy: {lstm_result['accuracy']*100:.2f}%")
    logger.info(f"  Gespeichert: models/lstm_best.pt")

    logger.info("\nRL Agent (PPO):")
    logger.info(f"  Return: {rl_metrics['total_return']*100:.2f}%")
    logger.info(f"  Sharpe: {rl_metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Gespeichert: models/ppo_trading.zip")

    logger.info("\n" + "=" * 60)
    logger.info("Training abgeschlossen!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
