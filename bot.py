import os
import time
import requests
import pandas as pd
import numpy as np
from binance.client import Client
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from prophet import Prophet
import datetime
import telegram

# Load env variables
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

client = Client(API_KEY, API_SECRET)

# Telegram bot setup (optional)
bot = None
if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
    bot = telegram.Bot(token=TELEGRAM_TOKEN)

def send_telegram(message):
    if bot:
        try:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        except Exception as e:
            print(f"Telegram send error: {e}")

# Constants
SYMBOL = "BTCUSDT"
INTERVAL = "1d"
LOOKBACK_DAYS = 90
INVESTMENT_USDT = 100  # Amount per trade
STOP_LOSS_PERCENT = 0.05  # 5% stop loss
TAKE_PROFIT_PERCENT = 0.10  # 10% take profit

def get_historical_klines(symbol, interval, lookback):
    """Fetch historical klines from Binance"""
    klines = client.get_historical_klines(symbol, interval, f"{lookback} day ago UTC")
    data = []
    for k in klines:
        data.append({
            "open_time": datetime.datetime.fromtimestamp(k[0] / 1000),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5])
        })
    return pd.DataFrame(data)

def forecast_price(df):
    df_prophet = df[['open_time', 'close']].rename(columns={'open_time': 'ds', 'close': 'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(7)

def calculate_indicators(df):
    rsi = RSIIndicator(df['close']).rsi()
    sma = SMAIndicator(df['close']).sma_indicator()
    df['RSI'] = rsi
    df['SMA'] = sma
    return df

def get_balance(asset="USDT"):
    balance = client.get_asset_balance(asset=asset)
    if balance:
        return float(balance['free'])
    return 0.0

def place_order(symbol, side, quantity):
    try:
        order = client.create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=quantity
        )
        send_telegram(f"Order executed: {side} {quantity} {symbol}")
        return order
    except Exception as e:
        send_telegram(f"Order error: {e}")
        return None

def get_quantity(symbol, usdt_amount):
    price = float(client.get_symbol_ticker(symbol=symbol)['price'])
    qty = usdt_amount / price
    # Binance requires quantity to be rounded to correct step size
    info = client.get_symbol_info(symbol)
    step_size = 0.000001
    for f in info['filters']:
        if f['filterType'] == 'LOT_SIZE':
            step_size = float(f['stepSize'])
            break
    qty = np.floor(qty / step_size) * step_size
    return round(qty, 6)

def main():
    send_telegram("Bot started")
    while True:
        df = get_historical_klines(SYMBOL, INTERVAL, LOOKBACK_DAYS)
        df = calculate_indicators(df)
        forecast = forecast_price(df)
        
        current_price = df['close'].iloc[-1]
        forecasted_price_avg = forecast['yhat'].mean()
        current_rsi = df['RSI'].iloc[-1]
        current_sma = df['SMA'].iloc[-1]

        send_telegram(f"Price: {current_price:.2f}, Forecast Avg: {forecasted_price_avg:.2f}, RSI: {current_rsi:.2f}, SMA: {current_sma:.2f}")

        usdt_balance = get_balance("USDT")
        btc_balance = get_balance("BTC")

        # Buy logic: strong buy signal, enough USDT balance
        if current_rsi < 30 and forecasted_price_avg > current_price * 1.07 and usdt_balance >= INVESTMENT_USDT:
            qty = get_quantity(SYMBOL, INVESTMENT_USDT)
            order = place_order(SYMBOL, "BUY", qty)

        # Sell logic: price above take profit or RSI high, and holding BTC
        elif (current_price > forecasted_price_avg * (1 + TAKE_PROFIT_PERCENT) or current_rsi > 70) and btc_balance > 0:
            order = place_order(SYMBOL, "SELL", btc_balance)

        # Stop loss logic: if price drops below 5% from purchase price - needs tracking (simplified here)
        # You can expand with order book / purchase history tracking

        else:
            print("No action taken.")

        time.sleep(60*60)  # Run hourly

if __name__ == "__main__":
    main()
