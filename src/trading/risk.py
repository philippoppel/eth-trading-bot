"""
Risikomanagement Modul.
"""

import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger


@dataclass
class RiskState:
    """Aktueller Risiko-Status."""
    can_trade: bool = True
    reason: str = ""
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    position_size_multiplier: float = 1.0


class RiskManager:
    """Risikomanagement für Trading."""

    def __init__(
        self,
        initial_balance: float = 10000.0,
        max_position_pct: float = 0.30,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        daily_loss_limit_pct: float = 0.03,
        max_drawdown_pct: float = 0.10,
        max_consecutive_losses: int = 5,
        cooldown_minutes: int = 60
    ):
        """
        Args:
            initial_balance: Startkapital
            max_position_pct: Max. Position in % des Kapitals
            stop_loss_pct: Stop Loss Prozent
            take_profit_pct: Take Profit Prozent
            daily_loss_limit_pct: Max. Tagesverlust
            max_drawdown_pct: Max. Drawdown (Kill Switch)
            max_consecutive_losses: Max. Verluste in Folge
            cooldown_minutes: Pause nach Verlustserie
        """
        self.initial_balance = initial_balance
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_minutes = cooldown_minutes

        # State
        self.peak_balance = initial_balance
        self.current_balance = initial_balance
        self.daily_start_balance = initial_balance
        self.last_daily_reset = datetime.now().date()
        self.consecutive_losses = 0
        self.cooldown_until: Optional[datetime] = None
        self.trade_history: List[dict] = []
        self.is_killed = False

    def update_balance(self, new_balance: float):
        """Aktualisiert Balance und Drawdown."""
        self.current_balance = new_balance

        # Peak aktualisieren
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

        # Daily Reset prüfen
        today = datetime.now().date()
        if today > self.last_daily_reset:
            self.daily_start_balance = new_balance
            self.last_daily_reset = today
            logger.info(f"Daily Reset: Start Balance = ${new_balance:.2f}")

    def record_trade(self, pnl: float):
        """Zeichnet Trade auf und aktualisiert Verlust-Counter."""
        self.trade_history.append({
            'time': datetime.now(),
            'pnl': pnl
        })

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Cooldown aktivieren bei zu vielen Verlusten
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.cooldown_until = datetime.now() + timedelta(minutes=self.cooldown_minutes)
            logger.warning(
                f"Cooldown aktiviert: {self.consecutive_losses} Verluste in Folge. "
                f"Pause bis {self.cooldown_until}"
            )

    def check_risk(self) -> RiskState:
        """
        Prüft alle Risiko-Bedingungen.

        Returns:
            RiskState mit Trading-Erlaubnis und Details
        """
        state = RiskState()

        # Kill Switch aktiv?
        if self.is_killed:
            state.can_trade = False
            state.reason = "Kill Switch aktiviert"
            return state

        # Cooldown aktiv?
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            state.can_trade = False
            state.reason = f"Cooldown bis {self.cooldown_until.strftime('%H:%M')}"
            state.consecutive_losses = self.consecutive_losses
            return state

        # Reset Cooldown wenn abgelaufen
        if self.cooldown_until and datetime.now() >= self.cooldown_until:
            self.cooldown_until = None
            self.consecutive_losses = 0
            logger.info("Cooldown beendet")

        # Max Drawdown prüfen (Kill Switch)
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        state.current_drawdown = current_drawdown

        if current_drawdown >= self.max_drawdown_pct:
            self.is_killed = True
            state.can_trade = False
            state.reason = f"Kill Switch: Max Drawdown erreicht ({current_drawdown*100:.1f}%)"
            logger.error(state.reason)
            return state

        # Daily Loss Limit prüfen
        daily_pnl = (self.current_balance - self.daily_start_balance) / self.daily_start_balance
        state.daily_pnl = daily_pnl

        if daily_pnl <= -self.daily_loss_limit_pct:
            state.can_trade = False
            state.reason = f"Daily Loss Limit erreicht ({daily_pnl*100:.1f}%)"
            logger.warning(state.reason)
            return state

        # Consecutive Losses prüfen
        state.consecutive_losses = self.consecutive_losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            state.can_trade = False
            state.reason = f"Zu viele Verluste in Folge ({self.consecutive_losses})"
            return state

        # Position Size Multiplier basierend auf Drawdown
        if current_drawdown > 0.05:
            state.position_size_multiplier = 0.5  # Reduzierte Position bei >5% DD
        elif current_drawdown > 0.03:
            state.position_size_multiplier = 0.75  # Leicht reduziert bei >3% DD
        else:
            state.position_size_multiplier = 1.0

        state.can_trade = True
        return state

    def calculate_position_size(
        self,
        current_price: float,
        volatility: float = None
    ) -> float:
        """
        Berechnet optimale Position Size.

        Args:
            current_price: Aktueller Preis
            volatility: ATR oder ähnliches Volatilitätsmaß

        Returns:
            Position Size in Basiswährung (z.B. ETH)
        """
        risk_state = self.check_risk()

        if not risk_state.can_trade:
            return 0.0

        # Basis Position Size
        position_value = self.current_balance * self.max_position_pct

        # Volatilitäts-Anpassung
        if volatility and volatility > 0:
            # Bei hoher Volatilität kleinere Position
            vol_adjustment = min(1.0, 0.02 / volatility)  # Ziel: 2% Risiko
            position_value *= vol_adjustment

        # Drawdown Multiplier anwenden
        position_value *= risk_state.position_size_multiplier

        # In Basiswährung umrechnen
        position_size = position_value / current_price

        return position_size

    def calculate_stop_loss(self, entry_price: float, side: str = 'long') -> float:
        """Berechnet Stop Loss Preis."""
        if side == 'long':
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)

    def calculate_take_profit(self, entry_price: float, side: str = 'long') -> float:
        """Berechnet Take Profit Preis."""
        if side == 'long':
            return entry_price * (1 + self.take_profit_pct)
        else:
            return entry_price * (1 - self.take_profit_pct)

    def reset_kill_switch(self):
        """Reset Kill Switch (manuell)."""
        self.is_killed = False
        self.consecutive_losses = 0
        self.cooldown_until = None
        logger.info("Kill Switch zurückgesetzt")

    def get_status(self) -> dict:
        """Gibt aktuellen Status zurück."""
        risk_state = self.check_risk()

        return {
            'can_trade': risk_state.can_trade,
            'reason': risk_state.reason,
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'current_drawdown': risk_state.current_drawdown,
            'daily_pnl': risk_state.daily_pnl,
            'consecutive_losses': risk_state.consecutive_losses,
            'position_size_multiplier': risk_state.position_size_multiplier,
            'is_killed': self.is_killed,
            'cooldown_until': self.cooldown_until
        }


if __name__ == "__main__":
    # Test
    risk = RiskManager(initial_balance=10000)

    # Simuliere Trades
    risk.update_balance(10000)
    print(f"Status: {risk.get_status()}")

    # Simuliere Verluste
    for i in range(6):
        risk.record_trade(-100)
        risk.update_balance(risk.current_balance - 100)
        status = risk.get_status()
        print(f"Trade {i+1}: Balance=${status['current_balance']:.2f}, "
              f"Can Trade: {status['can_trade']}, "
              f"Losses: {status['consecutive_losses']}")
