import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('preprocessed_crypto_data1.csv')

df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
df = df.sort_values(['Coin', 'Date'])

features = [
    'Open', 'High', 'Low', 'Volume', 'Marketcap', 'Daily_Return', 'MA_7', 'MA_30',
    'Close_lag1', 'Close_lag3', 'Close_lag7', 'Volume_lag1', 'Volume_lag3', 'Volume_lag7',
    'Daily_Return_lag1', 'Daily_Return_lag3', 'Daily_Return_lag7',
    'Price_Range', 'Close_std_7', 'RSI_14', 'BB_upper', 'BB_lower',
    'Month_sin', 'Month_cos', 'Weekday_sin', 'Weekday_cos'
]

def process_coin_data(coin_df):
    coin_df = coin_df.dropna(subset=['MA_7', 'MA_30']).copy()
    for lag in [1, 3, 7]:
        coin_df[f'Close_lag{lag}'] = coin_df['Close'].shift(lag)
        coin_df[f'Volume_lag{lag}'] = coin_df['Volume'].shift(lag)
        coin_df[f'Daily_Return_lag{lag}'] = coin_df['Daily_Return'].shift(lag)
    coin_df['Price_Range'] = coin_df['High'] - coin_df['Low']
    coin_df['Close_std_7'] = coin_df['Close'].rolling(window=7).std()
    coin_df['RSI_14'] = RSIIndicator(coin_df['Close'], window=14).rsi()
    bb = BollingerBands(coin_df['Close'], window=20, window_dev=2)
    coin_df['BB_upper'] = bb.bollinger_hband()
    coin_df['BB_lower'] = bb.bollinger_lband()
    coin_df['Month_sin'] = np.sin(2 * np.pi * coin_df['Month'] / 12)
    coin_df['Month_cos'] = np.cos(2 * np.pi * coin_df['Month'] / 12)
    coin_df['Weekday_sin'] = np.sin(2 * np.pi * coin_df['Weekday'] / 7)
    coin_df['Weekday_cos'] = np.cos(2 * np.pi * coin_df['Weekday'] / 7)
    coin_df = coin_df.dropna()
    return coin_df

def predict_and_plot_coin(coin_name):
    if not coin_name.startswith('coin_'):
        coin_name = 'coin_' + coin_name
    if coin_name not in df['Coin'].unique():
        print(f"Error: '{coin_name}' not found in the dataset. Available coins: {[c.replace('coin_', '') for c in df['Coin'].unique()]}")
        return
    print(f"\nProcessing {coin_name}...")
    coin_df = df[df['Coin'] == coin_name].copy()
    coin_df = process_coin_data(coin_df)
    if len(coin_df) < 30:
        print(f"Error: Insufficient data for {coin_name} ({len(coin_df)} rows)")
        return
    X = coin_df[features]
    y = coin_df['Close']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features, index=X.index)
    tscv = TimeSeriesSplit(n_splits=5)
    mae_scores, rmse_scores, mape_scores, dir_acc_scores = [], [], [], []
    for train_idx, test_idx in tscv.split(X_scaled):
        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        direction_accuracy = np.mean(np.sign(y_test[1:].values - y_test[:-1].values) == 
                                    np.sign(y_pred[1:] - y_pred[:-1])) * 100 if len(y_test) > 1 else 0
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        mape_scores.append(mape)
        dir_acc_scores.append(direction_accuracy)
    print(f"\nEvaluation Metrics for {coin_name}:")
    print(f"Average MAE: {np.mean(mae_scores):.2f}")
    print(f"Average RMSE: {np.mean(rmse_scores):.2f}")
    print(f"Average MAPE: {np.mean(mape_scores):.2f}%")
    print(f"Average Directional Accuracy: {np.mean(dir_acc_scores):.2f}%")
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_scaled, y)
    last_data = X_scaled.iloc[-1:].copy()
    next_pred = model.predict(last_data)[0]
    last_date = coin_df['Date'].iloc[-1]
    next_date = last_date + pd.Timedelta(days=1)
    last_close = coin_df['Close'].iloc[-1]
    print(f"\nPredicted Close for {coin_name} on {next_date.date()}: {next_pred:.2f}")
    print(f"Last Close on {last_date.date()}: {last_close:.2f}")
    feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print(f"\nFeature Importance for {coin_name}:")
    print(feature_importance.head(5))
    plt.figure(figsize=(12, 6))
    plt.plot(coin_df['Date'], coin_df['Close'], label='Historical Close', color='blue')
    plt.scatter([next_date], [next_pred], color='red', label='Predicted Close', zorder=5)
    plt.title(f'Close Prices for {coin_name}')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

results = []
predictions = []
for coin in df['Coin'].unique():
    print(f"Processing {coin}...")
    coin_df = df[df['Coin'] == coin].copy()
    coin_df = process_coin_data(coin_df)
    if len(coin_df) < 30:
        print(f"Skipping {coin} due to insufficient data ({len(coin_df)} rows)")
        continue
    X = coin_df[features]
    y = coin_df['Close']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features, index=X.index)
    tscv = TimeSeriesSplit(n_splits=5)
    mae_scores, rmse_scores, mape_scores, dir_acc_scores = [], [], [], []
    for train_idx, test_idx in tscv.split(X_scaled):
        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        direction_accuracy = np.mean(np.sign(y_test[1:].values - y_test[:-1].values) == 
                                    np.sign(y_pred[1:] - y_pred[:-1])) * 100 if len(y_test) > 1 else 0
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        mape_scores.append(mape)
        dir_acc_scores.append(direction_accuracy)
    results.append({
        'Coin': coin,
        'Average MAE': np.mean(mae_scores),
        'Average RMSE': np.mean(rmse_scores),
        'Average MAPE': np.mean(mape_scores),
        'Average Directional Accuracy': np.mean(dir_acc_scores)
    })
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_scaled, y)
    last_data = X_scaled.iloc[-1:].copy()
    next_pred = model.predict(last_data)[0]
    predictions.append({
        'Coin': coin,
        'Predicted Close': next_pred,
        'Last Date': coin_df['Date'].iloc[-1],
        'Last Close': coin_df['Close'].iloc[-1]
    })

results_df = pd.DataFrame(results)
results_df.to_csv('coin_prediction_metrics.csv', index=False)
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv('coin_predictions.csv', index=False)
print("\nEvaluation metrics saved to 'coin_prediction_metrics.csv'")
print("Predictions saved to 'coin_predictions.csv'")

while True:
    print("\nAvailable coins:", [c.replace('coin_', '') for c in df['Coin'].unique()])
    coin_input = input("Enter coin name (e.g., Bitcoin, Aave) or 'quit' to exit: ")
    if coin_input.lower() == 'quit':
        break
    predict_and_plot_coin(coin_input)