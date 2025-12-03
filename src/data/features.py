"""
Feature Engineering für Trading Bot.
Berechnet technische Indikatoren und statistische Features.
Verwendet die 'ta' Bibliothek (Technical Analysis Library).
"""

import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
from loguru import logger
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """Feature Engineering Pipeline."""

    def __init__(self):
        """Initialisiert Feature Engineer."""
        self.scaler = StandardScaler()
        self.feature_columns = []

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fügt technische Indikatoren hinzu.

        Args:
            df: DataFrame mit OHLCV Daten

        Returns:
            DataFrame mit zusätzlichen Indikator-Spalten
        """
        df = df.copy()

        # RSI
        df['rsi'] = momentum.RSIIndicator(df['close'], window=14).rsi()

        # MACD
        macd_indicator = trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_hist'] = macd_indicator.macd_diff()

        # Bollinger Bands
        bb = volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_pct'] = bb.bollinger_pband()

        # EMAs
        df['ema_9'] = trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema_21'] = trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['ema_50'] = trend.EMAIndicator(df['close'], window=50).ema_indicator()

        # EMA Crossovers
        df['ema_9_21_cross'] = (df['ema_9'] > df['ema_21']).astype(int)
        df['ema_21_50_cross'] = (df['ema_21'] > df['ema_50']).astype(int)

        # ATR (Volatilität)
        atr_indicator = volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        df['atr'] = atr_indicator.average_true_range()
        df['atr_pct'] = df['atr'] / df['close']

        # Volume SMA
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # ADX (Trend Stärke)
        adx_indicator = trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx_indicator.adx()
        df['di_plus'] = adx_indicator.adx_pos()
        df['di_minus'] = adx_indicator.adx_neg()

        # Stochastic
        stoch = momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # CCI
        df['cci'] = trend.CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()

        # Williams %R
        df['williams_r'] = momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14).williams_r()

        # OBV
        df['obv'] = volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['obv_norm'] = (df['obv'] - df['obv'].rolling(20).mean()) / df['obv'].rolling(20).std()

        logger.info(f"Technische Indikatoren hinzugefügt: {len(df.columns)} Spalten")

        return df

    def add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fügt statistische Features hinzu.

        Args:
            df: DataFrame mit OHLCV + Indikatoren

        Returns:
            DataFrame mit statistischen Features
        """
        df = df.copy()

        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Volatilität
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_20'] = df['returns'].rolling(20).std()

        # Z-Score des Preises
        df['price_zscore'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()

        # Momentum
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        df['momentum_20'] = df['close'].pct_change(20)

        # High-Low Range
        df['hl_range'] = (df['high'] - df['low']) / df['close']

        # Candle Body
        df['body'] = (df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']

        logger.info(f"Statistische Features hinzugefügt: {len(df.columns)} Spalten")

        return df

    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fügt Marktregime-Features hinzu.

        Args:
            df: DataFrame

        Returns:
            DataFrame mit Regime-Features
        """
        df = df.copy()

        # Trend Regime (basierend auf EMA Ausrichtung)
        df['trend_regime'] = 0
        df.loc[(df['ema_9'] > df['ema_21']) & (df['ema_21'] > df['ema_50']), 'trend_regime'] = 1  # Uptrend
        df.loc[(df['ema_9'] < df['ema_21']) & (df['ema_21'] < df['ema_50']), 'trend_regime'] = -1  # Downtrend

        # Volatilitätsregime
        vol_median = df['volatility_20'].median()
        df['vol_regime'] = (df['volatility_20'] > vol_median).astype(int)

        # ADX Regime (Trend vs Range)
        df['adx_regime'] = (df['adx'] > 25).astype(int)  # 1 = Trending, 0 = Ranging

        logger.info(f"Regime Features hinzugefügt: {len(df.columns)} Spalten")

        return df

    def process(self, df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
        """
        Vollständige Feature Pipeline.

        Args:
            df: Rohe OHLCV Daten
            normalize: Z-Score Normalisierung anwenden

        Returns:
            DataFrame mit allen Features
        """
        # Features hinzufügen
        df = self.add_technical_indicators(df)
        df = self.add_statistical_features(df)
        df = self.add_regime_features(df)

        # NaN entfernen (von Lookback-Perioden)
        df = df.dropna()

        # Feature-Spalten identifizieren
        exclude_cols = ['open', 'high', 'low', 'close', 'volume']
        self.feature_columns = [c for c in df.columns if c not in exclude_cols]

        # Normalisierung
        if normalize:
            df = self.normalize_features(df)

        logger.info(f"Feature Engineering abgeschlossen: {len(df)} Zeilen, {len(self.feature_columns)} Features")

        return df

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-Score Normalisierung der Features."""
        df = df.copy()

        # Nur Feature-Spalten normalisieren
        cols_to_normalize = [c for c in self.feature_columns if c in df.columns]

        for col in cols_to_normalize:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[f'{col}_norm'] = (df[col] - mean) / std
            else:
                df[f'{col}_norm'] = 0

        return df

    def get_feature_matrix(self, df: pd.DataFrame, normalized: bool = True) -> np.ndarray:
        """
        Extrahiert Feature-Matrix für Modelltraining.

        Args:
            df: Verarbeiteter DataFrame
            normalized: Normalisierte Features verwenden

        Returns:
            NumPy Array mit Features
        """
        if normalized:
            cols = [c for c in df.columns if c.endswith('_norm')]
        else:
            cols = self.feature_columns

        return df[cols].values


if __name__ == "__main__":
    # Test
    from binance_client import BinanceClient

    client = BinanceClient(testnet=True)
    df = client.get_historical_data('ETH/USDT', '15m', days=30)

    engineer = FeatureEngineer()
    df_processed = engineer.process(df)

    print(f"\nFeature Spalten: {len(engineer.feature_columns)}")
    print(f"Normalisierte Spalten: {len([c for c in df_processed.columns if c.endswith('_norm')])}")
    print(f"\nBeispiel Features:")
    print(df_processed[['close', 'rsi', 'macd', 'bb_pct', 'trend_regime']].tail())
