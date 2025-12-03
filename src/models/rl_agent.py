"""
Reinforcement Learning Agent für Trading.
Verwendet PPO aus stable-baselines3.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from loguru import logger
from pathlib import Path


class TradingEnv(gym.Env):
    """
    Trading Environment für Reinforcement Learning.

    Actions:
        0: Hold
        1: Buy
        2: Sell

    State:
        - Feature Vektor (technische Indikatoren)
        - Aktuelle Position (-1, 0, 1)
        - Unrealisierter PnL
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df,
        feature_columns: list,
        initial_balance: float = 10000.0,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        observation_window: int = 60,
        max_position: float = 1.0,
        reward_scaling: float = 1.0
    ):
        """
        Args:
            df: DataFrame mit OHLCV + Features
            feature_columns: Liste der Feature-Spalten
            initial_balance: Startkapital
            fee_rate: Trading Gebühren
            slippage_rate: Slippage
            observation_window: Lookback für State
            max_position: Maximale Position (1.0 = 100%)
            reward_scaling: Reward Skalierung
        """
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.observation_window = observation_window
        self.max_position = max_position
        self.reward_scaling = reward_scaling

        # Feature Matrix
        self.features = self.df[feature_columns].values
        self.prices = self.df['close'].values

        # State Dimension
        n_features = len(feature_columns)
        state_dim = n_features * observation_window + 2  # Features + Position + Unrealized PnL

        # Spaces
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

        # Initialisierung
        self.reset()

    def reset(self, seed=None, options=None):
        """Reset Environment."""
        super().reset(seed=seed)

        self.current_step = self.observation_window
        self.balance = self.initial_balance
        self.position = 0.0  # Anzahl ETH
        self.position_price = 0.0  # Durchschnittlicher Einstiegspreis
        self.portfolio_value = self.initial_balance
        self.trades = []
        self.history = []

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Erstellt Observation State."""
        # Feature Window
        start = self.current_step - self.observation_window
        end = self.current_step
        feature_window = self.features[start:end].flatten()

        # Position als normalisierter Wert
        position_normalized = self.position * self.prices[self.current_step] / self.portfolio_value

        # Unrealisierter PnL
        if self.position != 0:
            unrealized_pnl = (self.prices[self.current_step] - self.position_price) / self.position_price
        else:
            unrealized_pnl = 0.0

        # State zusammensetzen
        state = np.concatenate([
            feature_window,
            [position_normalized, unrealized_pnl]
        ])

        return state.astype(np.float32)

    def step(self, action):
        """
        Führt Action aus.

        Args:
            action: 0=Hold, 1=Buy, 2=Sell

        Returns:
            observation, reward, terminated, truncated, info
        """
        current_price = self.prices[self.current_step]

        # Action ausführen
        reward = 0.0
        trade_executed = False

        if action == 1 and self.position == 0:  # Buy
            # Kaufen mit verfügbarem Balance
            buy_amount = (self.balance * self.max_position) / current_price
            cost = buy_amount * current_price * (1 + self.fee_rate + self.slippage_rate)

            if cost <= self.balance:
                self.position = buy_amount
                self.position_price = current_price
                self.balance -= cost
                trade_executed = True
                self.trades.append({
                    'step': self.current_step,
                    'type': 'buy',
                    'price': current_price,
                    'amount': buy_amount
                })

        elif action == 2 and self.position > 0:  # Sell
            # Verkaufen
            sell_value = self.position * current_price * (1 - self.fee_rate - self.slippage_rate)
            profit_pct = (current_price - self.position_price) / self.position_price

            self.balance += sell_value
            self.trades.append({
                'step': self.current_step,
                'type': 'sell',
                'price': current_price,
                'amount': self.position,
                'profit_pct': profit_pct
            })

            # Reward basierend auf Trade-Profit
            reward = profit_pct * self.reward_scaling

            self.position = 0.0
            self.position_price = 0.0
            trade_executed = True

        # Portfolio Value aktualisieren
        self.portfolio_value = self.balance + self.position * current_price

        # Holding Reward (kleine Belohnung für profitable Positionen)
        if self.position > 0:
            unrealized_return = (current_price - self.position_price) / self.position_price
            reward += unrealized_return * 0.01  # Kleiner Holding-Bonus

        # History speichern
        self.history.append({
            'step': self.current_step,
            'price': current_price,
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': self.portfolio_value,
            'action': action
        })

        # Nächster Step
        self.current_step += 1

        # Episode Ende
        terminated = self.current_step >= len(self.prices) - 1
        truncated = False

        # Info
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'num_trades': len(self.trades),
            'balance': self.balance
        }

        return self._get_observation(), reward, terminated, truncated, info

    def get_metrics(self) -> dict:
        """Berechnet Trading-Metriken."""
        if not self.history:
            return {}

        portfolio_values = [h['portfolio_value'] for h in self.history]
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # Sharpe Ratio (annualisiert für 15m Daten)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 4)  # 15m -> annualisiert
        else:
            sharpe = 0

        # Max Drawdown
        peak = portfolio_values[0]
        max_dd = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        # Win Rate
        profitable_trades = [t for t in self.trades if t['type'] == 'sell' and t.get('profit_pct', 0) > 0]
        total_sells = [t for t in self.trades if t['type'] == 'sell']
        win_rate = len(profitable_trades) / len(total_sells) if total_sells else 0

        # Total Return
        total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'num_trades': len(self.trades),
            'final_value': self.portfolio_value
        }


def create_ppo_agent(
    env: TradingEnv,
    learning_rate: float = 0.0003,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    verbose: int = 1
) -> PPO:
    """
    Erstellt PPO Agent.

    Returns:
        PPO Agent
    """
    # Vectorized Environment
    vec_env = DummyVecEnv([lambda: env])

    agent = PPO(
        policy='MlpPolicy',
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=verbose
    )

    logger.info("PPO Agent erstellt")

    return agent


def train_ppo(
    train_env: TradingEnv,
    eval_env: TradingEnv = None,
    total_timesteps: int = 100000,
    save_path: str = 'models/ppo_trading'
) -> PPO:
    """
    Trainiert PPO Agent.

    Returns:
        Trainierter PPO Agent
    """
    # Agent erstellen
    agent = create_ppo_agent(train_env)

    # Callbacks
    callbacks = []

    if eval_env:
        eval_callback = EvalCallback(
            DummyVecEnv([lambda: eval_env]),
            best_model_save_path=str(Path(save_path).parent / 'best'),
            log_path='./logs/eval/',
            eval_freq=10000,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)

    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=str(Path(save_path).parent / 'checkpoints'),
        name_prefix='ppo'
    )
    callbacks.append(checkpoint_callback)

    # Training
    logger.info(f"Starte PPO Training für {total_timesteps} Timesteps...")

    agent.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=False
    )

    # Speichern
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(save_path)
    logger.info(f"Agent gespeichert: {save_path}")

    return agent


def load_ppo_agent(path: str, env: TradingEnv = None) -> PPO:
    """Lädt trainierten Agent."""
    if env:
        vec_env = DummyVecEnv([lambda: env])
        agent = PPO.load(path, env=vec_env)
    else:
        agent = PPO.load(path)

    logger.info(f"Agent geladen: {path}")
    return agent


if __name__ == "__main__":
    # Test mit Dummy-Daten
    import pandas as pd

    # Dummy DataFrame
    n = 5000
    df = pd.DataFrame({
        'close': 2000 + np.cumsum(np.random.randn(n) * 10),
        'feature1_norm': np.random.randn(n),
        'feature2_norm': np.random.randn(n),
        'feature3_norm': np.random.randn(n),
    })

    feature_cols = ['feature1_norm', 'feature2_norm', 'feature3_norm']

    # Environment
    env = TradingEnv(df, feature_cols, observation_window=60)

    # Test Episode
    obs, _ = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    metrics = env.get_metrics()
    print(f"Metrics: {metrics}")
