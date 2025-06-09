# CryptoMarketTrendAnalysis

This project uses machine learning to analyze and predict trends in the cryptocurrency market based on historical price data.

## 📂 Dataset

The dataset used for this project is the [Cryptocurrency Price History](https://www.kaggle.com/datasets/malikhasanali/cryptomarkettrendanalysis) by Hasan ALi, available on Kaggle.

To download the dataset, run the following code on your local system:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("malikhasanali/cryptomarkettrendanalysis")

print("Path to dataset files:", path)

# Cryptocurrency Price Prediction

This project predicts the next day's closing price for cryptocurrencies using the `preprocessed_crypto_data.csv` dataset. The Python script employs an XGBoost model trained on features like lagged prices, technical indicators (RSI, Bollinger Bands), and cyclical date encodings. It supports predictions for all coins in the dataset and includes an interactive function allowing users to input a coin name (e.g., "Bitcoin" for `coin_Bitcoin`) to view its predicted price, evaluation metrics, feature importance, and a chart of historical and predicted prices.

## Features
- **Multi-Coin Predictions**: Trains an XGBoost model for each coin, saving metrics (MAE, RMSE, MAPE, directional accuracy) and predictions to CSV files.
- **Feature Engineering**: Generates lagged features, volatility measures, RSI, Bollinger Bands, and cyclical encodings for month and weekday.
- **Interactive Selection**: Users can input a coin name (e.g., "Bitcoin") to view its predicted closing price, metrics, and a matplotlib chart.
- **Time-Series Cross-Validation**: Uses 5-fold time-series splits to evaluate model performance, respecting temporal order.
- **Output**: Saves results to `coin_prediction_metrics.csv` (metrics) and `coin_predictions.csv` (predictions).

## Requirements
- Python 3.8+
- Libraries:
  ```bash
  pip install pandas numpy scikit-learn xgboost ta matplotlib
