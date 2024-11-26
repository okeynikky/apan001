import os
import sys  # Import sys to access argv
import gym
from gym import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import joblib
from torch import nn
import asyncio
import websockets
import json
import time
import requests
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import mplfinance as mpf

# Custom Neural Network Architecture
class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim=256)
        n_input_channels = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(n_input_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)

# Indicator functions
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(series, fastperiod=12, slowperiod=26, signalperiod=9):
    ema_fast = ema(series, fastperiod)
    ema_slow = ema(series, slowperiod)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signalperiod)
    return macd_line, signal_line

def calculate_indicators(data):
    data['RSI'] = rsi(data['Close'], period=14)
    data['EMA_Fast'] = ema(data['Close'], period=3)
    data['EMA_Slow'] = ema(data['Close'], period=9)
    data['MACD'], data['MACD_Signal'] = macd(data['Close'])
    envelope_length = 21
    envelope_percent = 0.3 / 100
    data['Envelope_Upper'] = ema(data['Close'], period=envelope_length) * (1 + envelope_percent)
    data['Envelope_Lower'] = ema(data['Close'], period=envelope_length) * (1 - envelope_percent)
    return data

def calculate_scores(data):
    data = data.reset_index(drop=True)
    data['score_ema'] = 0
    data['score_macd'] = 0
    data['score_rsi30'] = 0
    data['score_envelope'] = 0
    data['sell_score_ema'] = 0
    data['sell_score_macd'] = 0
    data['sell_score_rsi70'] = 0
    data['sell_score_envelope'] = 0
    max_buy_score = 400  # Adjusted because there are 4 buy score components
    max_sell_score = 400  # Adjusted because there are 4 sell score components

    for i in range(1, len(data)):
        # Buy scores
        if (data['EMA_Fast'].iloc[i] > data['EMA_Slow'].iloc[i] and
            data['EMA_Fast'].iloc[i-1] <= data['EMA_Slow'].iloc[i-1]):
            data.at[i, 'score_ema'] = 100
        else:
            data.at[i, 'score_ema'] = max(0, data.iloc[i-1]['score_ema'] - 10)

        if (data['MACD'].iloc[i] > data['MACD_Signal'].iloc[i] and
            data['MACD'].iloc[i] < 0 and
            data['MACD'].iloc[i-1] <= data['MACD_Signal'].iloc[i-1]):
            data.at[i, 'score_macd'] = 100
        else:
            data.at[i, 'score_macd'] = max(0, data.iloc[i-1]['score_macd'] - 10)

        if (data['RSI'].iloc[i] > 30 and
            data['RSI'].iloc[i-1] <= 30):
            data.at[i, 'score_rsi30'] = 100
        else:
            data.at[i, 'score_rsi30'] = max(0, data.iloc[i-1]['score_rsi30'] - 10)

        if (data['EMA_Fast'].iloc[i] > data['Envelope_Lower'].iloc[i] and
            data['EMA_Fast'].iloc[i-1] <= data['Envelope_Lower'].iloc[i-1]):
            data.at[i, 'score_envelope'] = 100
        else:
            data.at[i, 'score_envelope'] = max(0, data.iloc[i-1]['score_envelope'] - 10)

        # Sell scores
        if (data['EMA_Fast'].iloc[i] < data['EMA_Slow'].iloc[i] and
            data['EMA_Fast'].iloc[i-1] >= data['EMA_Slow'].iloc[i-1]):
            data.at[i, 'sell_score_ema'] = 100
        else:
            data.at[i, 'sell_score_ema'] = max(0, data.iloc[i-1]['sell_score_ema'] - 10)

        if (data['MACD'].iloc[i] < data['MACD_Signal'].iloc[i] and
            data['MACD'].iloc[i] > 0 and
            data['MACD'].iloc[i-1] >= data['MACD_Signal'].iloc[i-1]):
            data.at[i, 'sell_score_macd'] = 100
        else:
            data.at[i, 'sell_score_macd'] = max(0, data.iloc[i-1]['sell_score_macd'] - 10)

        if (data['RSI'].iloc[i] < 70 and
            data['RSI'].iloc[i-1] >= 70):
            data.at[i, 'sell_score_rsi70'] = 100
        else:
            data.at[i, 'sell_score_rsi70'] = max(0, data.iloc[i-1]['sell_score_rsi70'] - 10)

        if (data['EMA_Fast'].iloc[i] < data['Envelope_Upper'].iloc[i] and
            data['EMA_Fast'].iloc[i-1] >= data['Envelope_Upper'].iloc[i-1]):
            data.at[i, 'sell_score_envelope'] = 100
        else:
            data.at[i, 'sell_score_envelope'] = max(0, data.iloc[i-1]['sell_score_envelope'] - 10)

    data['score'] = data['score_ema'] + data['score_macd'] + data['score_rsi30'] + data['score_envelope']
    data['sell_score'] = data['sell_score_ema'] + data['sell_score_macd'] + data['sell_score_rsi70'] + data['sell_score_envelope']
    data['total_percent'] = (data['score'] / max_buy_score) * 100
    data['total_sell_percent'] = (data['sell_score'] / max_sell_score) * 100

    return data

# Function to create and send order messages
def create_message(stockName, price, takeProfit, stopLoss, qty, side):
    message = [{
        'botId': 'robot.osai_apan_001@techglobetrading.com',
        'stockName': str(stockName),
        'timestamp': str(pd.Timestamp(datetime.now().strftime("%Y-%m-%d %H:%M:%S")).tz_localize("US/Eastern")),
        'price': float(price),
        'takeProfit': float(takeProfit),
        'stopLoss': float(stopLoss),
        'qty': float(qty),
        'strategy': 'apan001',
        'side': str(side),
        'type': 'market',
        'broker': 'alpaca'
    }]

    url = 'https://robot.techglobetrading.com/api/createorder'

    for m in message:
        response = requests.post(url, data=m)
        if response.status_code == 200:
            print(" [/] Sent %r" % m)
        else:
            print(" [x] Failed to send %r" % m)
            print(" [x] Response: %s" % response.text)

# Function to execute trades
def execute_trade(action, price, qty, symbol):
    # Define take profit and stop loss percentages
    target_profit_percentage = 0.05  # 5% profit target
    stop_loss_percentage = 0.03      # 3% stop loss

    if action == 2:  # Buy
        side = 'buy'
    elif action == 0:  # Sell
        side = 'sell'
    else:
        return  # No action needed for hold

    # Calculate take profit and stop loss prices
    if side == 'buy':
        take_profit = price * (1 + target_profit_percentage)
        stop_loss = price * (1 - stop_loss_percentage)
    elif side == 'sell':
        take_profit = price * (1 - target_profit_percentage)
        stop_loss = price * (1 + stop_loss_percentage)

    # Create and send the order message
    create_message(
        stockName=symbol,
        price=price,
        takeProfit=take_profit,
        stopLoss=stop_loss,
        qty=qty,
        side=side
    )

# Modified Plotting function
def plot_signals(data, fig, axlist):
    data = data.copy()
    data.set_index('timestamp', inplace=True)
    plot_data = data[['open', 'high', 'low', 'close', 'volume']]
    plot_data = plot_data.dropna(subset=['low', 'high', 'open', 'close', 'volume'])

    buy_signals = data[data['Buy_Signal']]
    sell_signals = data[data['Sell_Signal']]

    # Prepare markers for buy and sell signals
    apds = []

    # Initialize marker series with NaNs, indexed by plot_data.index
    buy_marker = pd.Series(np.nan, index=plot_data.index)
    sell_marker = pd.Series(np.nan, index=plot_data.index)

    # Set values where there are buy/sell signals
    if not buy_signals.empty:
        # Use the 'close' price for buy markers
        buy_marker.loc[buy_signals.index] = buy_signals['close']

    if not sell_signals.empty:
        # Use the 'close' price for sell markers
        sell_marker.loc[sell_signals.index] = sell_signals['close']

    # Create addplots if markers are not all NaN
    if not buy_marker.isna().all():
        apds.append(mpf.make_addplot(
            buy_marker,
            type='scatter',
            markersize=100,
            marker='^',
            color='green',
            ax=axlist[0]  # Use the main price axis
        ))

    if not sell_marker.isna().all():
        apds.append(mpf.make_addplot(
            sell_marker,
            type='scatter',
            markersize=100,
            marker='v',
            color='red',
            ax=axlist[0]  # Use the main price axis
        ))

    # Clear the axes
    for ax in axlist:
        ax.clear()

    # Plot on the existing axes
    mpf.plot(
        plot_data,
        type='candle',
        addplot=apds,
        ax=axlist[0],
        volume=axlist[1],
        style='charles',
        warn_too_much_data=2000,
        show_nontrading=True
    )

    # Add timestamp to the plot title
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    axlist[0].set_title(f'Trading Position as of {timestamp}')

    # Redraw the canvas
    plt.draw()
    plt.pause(0.01)

    # Get the script name without extension
    script_name = os.path.basename(sys.argv[0])  # e.g., "my_script.py"
    script_name_no_ext = os.path.splitext(script_name)[0]  # e.g., "my_script"

    # Save the figure to a file with script name
    filename = f"trading_position_train_{script_name_no_ext}.png"
    plt.savefig(filename)

# Trading Environment for Real-Time Data
class TradingEnv(gym.Env):
    def __init__(self, initial_balance=10000, max_stock=100, technical_indicators=True, scaler=None, symbol='TQQQ', live_trade=False):
        super(TradingEnv, self).__init__()
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_stock = max_stock
        self.stock_held = 0
        self.technical_indicators = technical_indicators
        self.transaction_cost = 0.0001
        self.cost_basis = 0.0
        self.trade_history = []
        self.position_returns = []
        self.current_step = 0  # For compatibility
        self.symbol = symbol  # Stock symbol
        self.live_trade = live_trade

        # For storing historical data
        self.data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'Buy_Signal', 'Sell_Signal'])
        self.tech_ind_columns = ['RSI', 'MACD', 'MACD_Signal', 'score', 'sell_score', 'total_percent', 'total_sell_percent']
        self.indicators_history = pd.DataFrame(columns=self.tech_ind_columns)

        self.scaler = scaler  # Load the scaler used during training

        # Observation and action spaces
        num_indicators = len(self.tech_ind_columns) if self.technical_indicators else 0
        obs_shape = 2 + num_indicators + 2  # Close price, scaled Volume, technical indicators, balance, stock_held
        obs_low = -np.inf * np.ones(obs_shape)
        obs_high = np.inf * np.ones(obs_shape)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # Sell, Hold, Buy

        # Capture the execution timestamp and set the output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_filename = f"output_{timestamp}.csv"

    def step(self, action):
        # Implement your step logic
        current_price = self.data['close'].iloc[-1]
        current_date = self.data['timestamp'].iloc[-1]
        reward = 0

        trade_executed = False  # Flag to check if a trade was executed

        # Execute the action
        if action == 2:  # Buy
            max_buyable = int(self.balance / (current_price * (1 + self.transaction_cost)))
            buy_amount = min(max_buyable, self.max_stock - self.stock_held)
            if buy_amount > 0:
                total_cost = buy_amount * current_price * (1 + self.transaction_cost)
                self.balance -= total_cost
                self.stock_held += buy_amount
                self.cost_basis = ((self.cost_basis * (self.stock_held - buy_amount)) + (current_price * buy_amount)) / self.stock_held
                # Record trade
                self.trade_history.append({
                    'step': self.current_step,
                    'type': 'buy',
                    'amount': buy_amount,
                    'price': current_price,
                    'date': current_date
                })
                trade_executed = True  # Trade was executed
                # Save trade history
                self.save_trade_history()
                # Mark Buy_Signal
                self.data.at[self.data.index[-1], 'Buy_Signal'] = True
                # Execute the trade
                if self.live_trade:
                    execute_trade(action=2, price=current_price, qty=buy_amount, symbol=self.symbol)
        elif action == 0:  # Sell
            if self.stock_held > 0:
                sell_amount = self.stock_held
                total_sale = sell_amount * current_price * (1 - self.transaction_cost)
                self.balance += total_sale
                position_return = (total_sale - (sell_amount * self.cost_basis)) / (sell_amount * self.cost_basis)
                self.position_returns.append(position_return)
                self.stock_held = 0
                self.cost_basis = 0
                # Record trade
                self.trade_history.append({
                    'step': self.current_step,
                    'type': 'sell',
                    'amount': sell_amount,
                    'price': current_price,
                    'date': current_date
                })
                trade_executed = True  # Trade was executed
                # Save trade history
                self.save_trade_history()
                # Mark Sell_Signal
                self.data.at[self.data.index[-1], 'Sell_Signal'] = True
                # Execute the trade
                if self.live_trade:
                    execute_trade(action=0, price=current_price, qty=sell_amount, symbol=self.symbol)

        # Update current step
        self.current_step += 1

        # Calculate portfolio value
        portfolio_value = self.balance + self.stock_held * current_price
        reward += (portfolio_value - self.initial_balance) / self.initial_balance

        # Prepare the next observation
        obs = self._get_observation()
        done = False  # In live trading, done is usually False
        info = {}

        return obs, reward, done, info

    # Method to save trade history
    def save_trade_history(self):
        df = pd.DataFrame(self.trade_history)
        df.to_csv(self.output_filename, index=False)

    def reset(self):
        # Reset the environment's state
        self.balance = self.initial_balance
        self.stock_held = 0
        self.cost_basis = 0.0
        self.trade_history = []
        self.position_returns = []
        self.data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'Buy_Signal', 'Sell_Signal'])
        self.indicators_history = pd.DataFrame(columns=self.tech_ind_columns)
        self.current_step = 0
        return self._get_observation()

    def _get_observation(self):
        # Get the current step
        obs = []

        # Current Close price and scaled Volume
        current_price = self.data['close'].iloc[-1]  # 'Close' price, unscaled
        current_volume = self.data['volume'].iloc[-1]  # 'Volume', scaled
        obs.extend([current_price, current_volume])

        # Technical indicators at current time
        if self.technical_indicators and len(self.indicators_history) > 0:
            current_indicators = self.indicators_history.iloc[-1][self.tech_ind_columns].values
            obs.extend(current_indicators)
        else:
            obs.extend([0]*len(self.tech_ind_columns))

        # Add normalized balance and stock held
        obs.extend([self.balance / self.initial_balance, self.stock_held / self.max_stock])

        return np.array(obs, dtype=np.float32)

    def update_data(self, new_data):
        # Append new data to self.data DataFrame
        new_row = {
            'timestamp': new_data['timestamp'],
            'open': new_data['open'],
            'high': new_data['high'],
            'low': new_data['low'],
            'close': new_data['close'],
            'volume': new_data['volume'],
            'Buy_Signal': False,
            'Sell_Signal': False
        }
        # Replace append with pd.concat
        new_row_df = pd.DataFrame([new_row])
        self.data = pd.concat([self.data, new_row_df], ignore_index=True)

        # Ensure 'timestamp' is of datetime type
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])

        # Scale 'Volume' using the scaler
        if self.scaler is not None:
            # Prepare data with zeros for 'Open', 'High', 'Low', and actual 'Volume'
            columns = ['Open', 'High', 'Low', 'Volume']  # Use the same column names as during fitting
            new_data_df = pd.DataFrame([[0, 0, 0, new_data['volume']]], columns=columns)
            scaled_data = self.scaler.transform(new_data_df)[0]
            self.data.at[self.data.index[-1], 'volume'] = scaled_data[3]  # 'Volume' is the fourth column

        # Ensure we have enough data to compute indicators
        if len(self.data) >= 2:  # At least 2 data points needed for indicators
            # Prepare data for indicators
            data_slice = self.data[-200:].copy()  # Use last 200 data points for stability
            data_slice.rename(columns={'close': 'Close', 'volume': 'Volume'}, inplace=True)

            data_slice = calculate_indicators(data_slice)
            data_slice = calculate_scores(data_slice)
            indicators_row = data_slice.iloc[-1][self.tech_ind_columns]

            # Concatenate the new row to the indicators_history DataFrame
            indicators_row_df = pd.DataFrame([indicators_row])
            self.indicators_history = pd.concat([self.indicators_history, indicators_row_df], ignore_index=True)
        else:
            # Append zeros if insufficient data
            indicators_row = pd.Series([0] * len(self.tech_ind_columns), index=self.tech_ind_columns)
            indicators_row_df = pd.DataFrame([indicators_row])
            self.indicators_history = pd.concat([self.indicators_history, indicators_row_df], ignore_index=True)

# Main function to run the live trading
def run_live_trading(model_filename, algo_class, symbol='TQQQ'):
    # Load the scaler used during training
    scaler = joblib.load('old/scaler_202406.save')  # Replace with your scaler filename if different

    # Initialize environment and model
    env = TradingEnv(
        initial_balance=80000,
        max_stock=100,
        technical_indicators=True,
        scaler=scaler,
        symbol=symbol,
        live_trade=True  # Set to True to enable live trading
    )

    # Load the trained model
    model = algo_class.load(
        model_filename,
        env=env,
        device='cpu',  # Change to 'cuda' if using a GPU
        custom_objects={
            'features_extractor_class': CustomFeatureExtractor
        }
    )

    # Initialize matplotlib interactive mode
    plt.ion()
    fig, axlist = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    plt.show()

    # Helper function to aggregate ticks into OHLCV bar
    def aggregate_ticks(ticks):
        # Assuming ticks is a list of dictionaries with keys: 'timestamp', 'price', 'volume'
        df = pd.DataFrame(ticks)
        if df.empty:
            return None

        aggregated_data = {
            'timestamp': df['timestamp'].iloc[-1],  # Use the last timestamp in the interval
            'open': df['price'].iloc[0],
            'high': df['price'].max(),
            'low': df['price'].min(),
            'close': df['price'].iloc[-1],
            'volume': df['volume'].sum()
        }
        return aggregated_data

    async def websocket_client():
        nonlocal fig, axlist  # Declare nonlocal to modify fig and axlist
        uri = "wss://data-socket.techglobetrading.com/"
        try:
            async with websockets.connect(uri) as websocket:
                print("Connected to server")

                ticks = []  # List to store incoming ticks for current interval
                interval_seconds = 15
                window_start_time = None

                while True:
                    try:
                        # Wait for new messages from the server
                        response = await websocket.recv()

                        data = json.loads(response)
                        # Extract data
                        if data['symbol'].upper() == symbol.upper():
                            tick_time = pd.to_datetime(data['end_timestamp'], unit='ms')

                            # Update this line to use the correct key
                            tick_price = float(data['close'])  # Use 'close' instead of 'price'

                            tick_volume = float(data['volume'])

                            # Initialize window_start_time if None
                            if window_start_time is None:
                                window_start_time = tick_time.floor(f'{interval_seconds}s')

                            # Check if the tick is within the current interval
                            if tick_time < window_start_time + pd.Timedelta(seconds=interval_seconds):
                                # Accumulate the tick data
                                ticks.append({
                                    'timestamp': tick_time,
                                    'price': tick_price,
                                    'volume': tick_volume
                                })
                            else:
                                # Aggregate the ticks into OHLCV bar
                                aggregated_data = aggregate_ticks(ticks)

                                if aggregated_data is not None:
                                    # Update the environment with the aggregated data
                                    env.update_data(aggregated_data)

                                    if len(env.data) >= 1:
                                        obs = env._get_observation()
                                        action, _ = model.predict(obs, deterministic=True)
                                        obs, reward, done, info = env.step(action)

                                        # Print or log the action
                                        print(f"Time: {aggregated_data['timestamp']}, Action: {action}, Price: {aggregated_data['close']}, Balance: {env.balance}, Stock Held: {env.stock_held}")

                                        # Plotting
                                        if len(env.data) > 10:
                                            plot_signals(env.data.copy(), fig, axlist)

                                # Reset the ticks list and window_start_time
                                ticks = [{
                                    'timestamp': tick_time,
                                    'price': tick_price,
                                    'volume': tick_volume
                                }]
                                window_start_time = tick_time.floor(f'{interval_seconds}s')

                        else:
                            # If data is for a different symbol, ignore it
                            pass

                    except websockets.ConnectionClosed:
                        print("Connection with server closed")
                        break
                    except Exception as e:
                        print(f"Error processing message: {e}")
                        # Optionally, log the data or stack trace for debugging
        except Exception as e:
            print(f"Error connecting to server: {e}")

    # Run the WebSocket client in the asyncio event loop
    asyncio.run(websocket_client())

# Entry point
if __name__ == "__main__":
    # Specify the model filename and algorithm class
    model_filename = 'PPO_Split1_202406.zip'  # Replace with your actual model filename
    algo_class = PPO  # Replace with your algorithm class (PPO or DQN)

    # Run live trading
    run_live_trading(model_filename, algo_class, symbol='TQQQ')