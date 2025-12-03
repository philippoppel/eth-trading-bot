"""
Binance Client für Testnet Trading.
Verwendet CCXT für API-Kommunikation.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()


class BinanceClient:
    """Binance Testnet Client für Daten und Trading."""

    def __init__(self, testnet: bool = True):
        """
        Initialisiert den Binance Client.

        Args:
            testnet: True für Testnet, False für Live
        """
        self.testnet = testnet

        if testnet:
            self.exchange = ccxt.binance({
                'apiKey': os.getenv('BINANCE_TESTNET_API_KEY'),
                'secret': os.getenv('BINANCE_TESTNET_API_SECRET'),
                'sandbox': True,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True
                }
            })
            logger.info("Binance TESTNET Client initialisiert")
        else:
            self.exchange = ccxt.binance({
                'apiKey': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_API_SECRET'),
                'enableRateLimit': True
            })
            logger.info("Binance LIVE Client initialisiert")

    def get_historical_data(
        self,
        symbol: str = 'ETH/USDT',
        timeframe: str = '15m',
        days: int = 180
    ) -> pd.DataFrame:
        """
        Lädt historische OHLCV Daten.

        Args:
            symbol: Trading Pair
            timeframe: Zeitrahmen (1m, 5m, 15m, 1h, 4h, 1d)
            days: Anzahl Tage zurück

        Returns:
            DataFrame mit OHLCV Daten
        """
        logger.info(f"Lade {days} Tage {timeframe} Daten für {symbol}...")

        # Berechne Start-Timestamp
        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        all_ohlcv = []

        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=1000
                )

                if not ohlcv:
                    break

                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1  # Nächster Timestamp

                # Prüfe ob wir am aktuellen Zeitpunkt sind
                if ohlcv[-1][0] > datetime.now().timestamp() * 1000 - 60000:
                    break

                logger.debug(f"Geladen: {len(all_ohlcv)} Kerzen")

            except Exception as e:
                logger.error(f"Fehler beim Laden: {e}")
                break

        if not all_ohlcv:
            raise ValueError("Keine Daten geladen!")

        # DataFrame erstellen
        df = pd.DataFrame(
            all_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()

        # Duplikate entfernen
        df = df[~df.index.duplicated(keep='first')]

        logger.info(f"Geladen: {len(df)} Kerzen von {df.index[0]} bis {df.index[-1]}")

        return df

    def get_current_price(self, symbol: str = 'ETH/USDT') -> float:
        """Holt aktuellen Preis."""
        ticker = self.exchange.fetch_ticker(symbol)
        return ticker['last']

    def get_balance(self, currency: str = 'USDT') -> float:
        """Holt verfügbares Guthaben."""
        balance = self.exchange.fetch_balance()
        return balance['free'].get(currency, 0)

    def place_market_order(
        self,
        symbol: str,
        side: str,
        amount: float
    ) -> dict:
        """
        Platziert eine Market Order.

        Args:
            symbol: Trading Pair
            side: 'buy' oder 'sell'
            amount: Menge in Basis-Währung (z.B. ETH)

        Returns:
            Order-Info Dictionary
        """
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount
            )
            logger.info(f"Order platziert: {side} {amount} {symbol} @ Market")
            return order
        except Exception as e:
            logger.error(f"Order fehlgeschlagen: {e}")
            raise

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float
    ) -> dict:
        """Platziert eine Limit Order."""
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=amount,
                price=price
            )
            logger.info(f"Limit Order: {side} {amount} {symbol} @ {price}")
            return order
        except Exception as e:
            logger.error(f"Order fehlgeschlagen: {e}")
            raise

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        """Storniert eine Order."""
        return self.exchange.cancel_order(order_id, symbol)

    def get_open_orders(self, symbol: str = None) -> list:
        """Holt offene Orders."""
        return self.exchange.fetch_open_orders(symbol)

    def save_data(self, df: pd.DataFrame, filename: str):
        """Speichert DataFrame als CSV."""
        path = Path('data') / f"{filename}.csv"
        path.parent.mkdir(exist_ok=True)
        df.to_csv(path)
        logger.info(f"Daten gespeichert: {path}")

    def load_data(self, filename: str) -> pd.DataFrame:
        """Lädt DataFrame aus CSV."""
        path = Path('data') / f"{filename}.csv"
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            logger.info(f"Daten geladen: {path}")
            return df
        return None


if __name__ == "__main__":
    # Test
    client = BinanceClient(testnet=True)

    # Historische Daten laden
    df = client.get_historical_data('ETH/USDT', '15m', days=30)
    print(df.tail())

    # Aktueller Preis
    price = client.get_current_price('ETH/USDT')
    print(f"Aktueller ETH Preis: ${price:.2f}")

    # Balance
    balance = client.get_balance('USDT')
    print(f"USDT Balance: ${balance:.2f}")
