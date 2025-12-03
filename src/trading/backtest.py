"""
Backtesting Engine für Trading-Strategien.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Callable
from loguru import logger


@dataclass
class Trade:
    """Einzelner Trade."""
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp = None
    exit_price: float = None
    side: str = 'long'  # 'long' oder 'short'
    size: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0


@dataclass
class BacktestResult:
    """Backtest Ergebnis."""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    num_trades: int = 0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    final_balance: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)


class Backtester:
    """Backtesting Engine."""

    def __init__(
        self,
        initial_balance: float = 10000.0,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        max_position_pct: float = 0.30
    ):
        """
        Args:
            initial_balance: Startkapital
            fee_rate: Trading Gebühren (0.1%)
            slippage_rate: Slippage (0.05%)
            stop_loss_pct: Stop Loss Prozent
            take_profit_pct: Take Profit Prozent
            max_position_pct: Max Position Size
        """
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_position_pct = max_position_pct

        # State
        self.balance = initial_balance
        self.position = 0.0
        self.position_price = 0.0
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.current_trade: Trade = None

    def reset(self):
        """Reset Backtester."""
        self.balance = self.initial_balance
        self.position = 0.0
        self.position_price = 0.0
        self.trades = []
        self.equity_curve = []
        self.current_trade = None

    def run(
        self,
        df: pd.DataFrame,
        signal_func: Callable[[pd.DataFrame, int], int]
    ) -> BacktestResult:
        """
        Führt Backtest durch.

        Args:
            df: DataFrame mit OHLCV Daten
            signal_func: Funktion die Signal zurückgibt (1=Buy, -1=Sell, 0=Hold)

        Returns:
            BacktestResult
        """
        self.reset()

        for i in range(len(df)):
            row = df.iloc[i]
            price = row['close']
            high = row['high']
            low = row['low']
            timestamp = df.index[i]

            # Check Stop Loss / Take Profit
            if self.position > 0:
                # Stop Loss
                if low <= self.position_price * (1 - self.stop_loss_pct):
                    self._close_position(
                        self.position_price * (1 - self.stop_loss_pct),
                        timestamp,
                        'stop_loss'
                    )
                # Take Profit
                elif high >= self.position_price * (1 + self.take_profit_pct):
                    self._close_position(
                        self.position_price * (1 + self.take_profit_pct),
                        timestamp,
                        'take_profit'
                    )

            # Signal generieren
            signal = signal_func(df, i)

            # Signal ausführen
            if signal == 1 and self.position == 0:  # Buy
                self._open_position(price, timestamp)
            elif signal == -1 and self.position > 0:  # Sell
                self._close_position(price, timestamp, 'signal')

            # Equity berechnen
            equity = self.balance + self.position * price
            self.equity_curve.append(equity)

        # Falls Position noch offen, schließen
        if self.position > 0:
            self._close_position(df.iloc[-1]['close'], df.index[-1], 'end')

        # Metriken berechnen
        return self._calculate_metrics()

    def _open_position(self, price: float, timestamp: pd.Timestamp):
        """Öffnet Position."""
        # Slippage
        entry_price = price * (1 + self.slippage_rate)

        # Position Size
        position_value = self.balance * self.max_position_pct
        self.position = position_value / entry_price

        # Fees
        fees = position_value * self.fee_rate
        self.balance -= (position_value + fees)

        self.position_price = entry_price

        self.current_trade = Trade(
            entry_time=timestamp,
            entry_price=entry_price,
            side='long',
            size=self.position,
            fees=fees
        )

    def _close_position(self, price: float, timestamp: pd.Timestamp, reason: str):
        """Schließt Position."""
        if self.position == 0:
            return

        # Slippage
        exit_price = price * (1 - self.slippage_rate)

        # Value
        exit_value = self.position * exit_price

        # Fees
        fees = exit_value * self.fee_rate
        self.balance += (exit_value - fees)

        # Trade abschließen
        if self.current_trade:
            self.current_trade.exit_time = timestamp
            self.current_trade.exit_price = exit_price
            self.current_trade.fees += fees
            self.current_trade.pnl = (exit_price - self.current_trade.entry_price) * self.current_trade.size - self.current_trade.fees
            self.current_trade.pnl_pct = (exit_price / self.current_trade.entry_price - 1) * 100
            self.trades.append(self.current_trade)

        self.position = 0.0
        self.position_price = 0.0
        self.current_trade = None

    def _calculate_metrics(self) -> BacktestResult:
        """Berechnet Backtest-Metriken."""
        result = BacktestResult()

        if not self.equity_curve:
            return result

        result.equity_curve = self.equity_curve
        result.trades = self.trades
        result.final_balance = self.balance
        result.num_trades = len(self.trades)

        # Total Return
        result.total_return = (self.balance - self.initial_balance) / self.initial_balance

        # Returns berechnen
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]

        # Sharpe Ratio (annualisiert für 15m)
        if len(returns) > 1 and np.std(returns) > 0:
            result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 4)
        else:
            result.sharpe_ratio = 0

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            result.sortino_ratio = np.mean(returns) / np.std(downside_returns) * np.sqrt(252 * 24 * 4)
        else:
            result.sortino_ratio = 0

        # Max Drawdown
        peak = equity[0]
        max_dd = 0
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        result.max_drawdown = max_dd

        # Trade Statistics
        if self.trades:
            pnls = [t.pnl for t in self.trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]

            result.win_rate = len(wins) / len(pnls) if pnls else 0
            result.avg_trade_pnl = np.mean(pnls) if pnls else 0
            result.avg_win = np.mean(wins) if wins else 0
            result.avg_loss = np.mean(losses) if losses else 0

            # Profit Factor
            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses)) if losses else 0
            result.profit_factor = total_wins / total_losses if total_losses > 0 else 0

        return result

    def print_results(self, result: BacktestResult):
        """Gibt Ergebnisse aus."""
        logger.info("=" * 50)
        logger.info("BACKTEST ERGEBNISSE")
        logger.info("=" * 50)
        logger.info(f"Total Return: {result.total_return*100:.2f}%")
        logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"Sortino Ratio: {result.sortino_ratio:.2f}")
        logger.info(f"Max Drawdown: {result.max_drawdown*100:.2f}%")
        logger.info(f"Win Rate: {result.win_rate*100:.2f}%")
        logger.info(f"Profit Factor: {result.profit_factor:.2f}")
        logger.info(f"Anzahl Trades: {result.num_trades}")
        logger.info(f"Avg Trade PnL: ${result.avg_trade_pnl:.2f}")
        logger.info(f"Final Balance: ${result.final_balance:.2f}")
        logger.info("=" * 50)


def simple_signal_func(df: pd.DataFrame, i: int) -> int:
    """
    Beispiel Signal-Funktion basierend auf RSI.

    Returns:
        1 = Buy, -1 = Sell, 0 = Hold
    """
    if i < 14:
        return 0

    rsi = df.iloc[i].get('rsi', 50)

    if rsi < 30:
        return 1  # Überverkauft -> Buy
    elif rsi > 70:
        return -1  # Überkauft -> Sell

    return 0


if __name__ == "__main__":
    # Test
    import sys
    sys.path.insert(0, '../..')

    from src.data.binance_client import BinanceClient
    from src.data.features import FeatureEngineer

    # Daten laden
    client = BinanceClient(testnet=True)
    df = client.get_historical_data('ETH/USDT', '15m', days=60)

    # Features
    engineer = FeatureEngineer()
    df = engineer.process(df, normalize=False)

    # Backtest
    backtester = Backtester()
    result = backtester.run(df, simple_signal_func)
    backtester.print_results(result)
