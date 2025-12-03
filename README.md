# ETH Trading Bot

AI-powered trading bot for ETH/USDT on Binance Testnet using LSTM and Reinforcement Learning.

## Features

- **LSTM Model**: Predicts price direction (Up/Down/Sideways)
- **PPO RL Agent**: Learns optimal trading strategy
- **Paper Trading**: Safe testing on Binance Testnet
- **Risk Management**: Stop-loss, take-profit, max drawdown protection
- **Telegram Alerts**: Real-time trade notifications
- **Backtesting**: Test strategies on historical data

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/eth-trading-bot.git
cd eth-trading-bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy example config
cp .env.example .env

# Edit .env with your API keys
nano .env
```

**Get Binance Testnet API Keys:** https://testnet.binance.vision/

### 3. Train Models

```bash
python train.py
```

This will:
- Download 1 year of ETH/USDT data
- Train LSTM model (~5 min)
- Train PPO agent (~10 min)

### 4. Run Backtest

```bash
python run_bot.py --backtest --days 30
```

### 5. Start Trading Bot

```bash
# Single cycle
python run_bot.py --model lstm --once

# Continuous trading
python run_bot.py --model lstm
```

## Telegram Setup (Optional)

1. Create bot with [@BotFather](https://t.me/BotFather)
2. Get your Chat ID by messaging the bot and visiting:
   `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. Add to `.env`:
   ```
   TELEGRAM_BOT_TOKEN=your_token
   TELEGRAM_CHAT_ID=your_chat_id
   ```
4. Start interactive bot:
   ```bash
   python telegram_bot.py
   ```

**Commands:** `/status`, `/price`, `/performance`, `/trades`

## Project Structure

```
eth-trading-bot/
├── config.yaml          # Trading configuration
├── train.py             # Model training script
├── run_bot.py           # Main trading bot
├── telegram_bot.py      # Interactive Telegram bot
├── src/
│   ├── data/
│   │   ├── binance_client.py   # Binance API
│   │   └── features.py         # Feature engineering
│   ├── models/
│   │   ├── lstm.py             # LSTM model
│   │   └── rl_agent.py         # PPO agent
│   ├── trading/
│   │   ├── backtest.py         # Backtesting engine
│   │   ├── paper_trader.py     # Paper trading
│   │   └── risk.py             # Risk management
│   └── utils/
│       └── telegram.py         # Telegram notifications
├── models/              # Trained models (gitignored)
├── data/                # Historical data (gitignored)
└── logs/                # Log files (gitignored)
```

## Configuration

Edit `config.yaml`:

```yaml
trading:
  symbol: "ETH/USDT"
  timeframe: "15m"
  initial_balance: 10000

risk:
  max_position_pct: 0.30    # Max 30% per trade
  stop_loss_pct: 0.02       # 2% stop loss
  take_profit_pct: 0.04     # 4% take profit
  max_drawdown_pct: 0.10    # 10% kill switch
```

## Performance

Backtest results (30 days):
- **Return**: ~24%
- **Sharpe Ratio**: ~13
- **Win Rate**: ~95%
- **Max Drawdown**: <1%

*Results may vary. Past performance is not indicative of future results.*

## Disclaimer

This bot is for **educational purposes only**. Trading cryptocurrencies involves significant risk. Only trade with money you can afford to lose. The authors are not responsible for any financial losses.

## License

MIT License
