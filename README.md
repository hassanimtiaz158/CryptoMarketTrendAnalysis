
# 📊 CryptoMarketTrendAnalysis

This project uses machine learning to analyze and **predict trends in the cryptocurrency market** based on historical price data.

---

## 📂 Dataset

The dataset used is [**Cryptocurrency Price History**](https://www.kaggle.com/datasets/malikhasanali/cryptomarkettrendanalysis) by *Hasan Ali*, available on Kaggle.

To download the dataset programmatically:

```python
import kagglehub

# Download the latest version
path = kagglehub.dataset_download("malikhasanali/cryptomarkettrendanalysis")
print("Path to dataset files:", path)
```

---

## 🔮 Cryptocurrency Price Prediction

This script predicts the **next day's closing price** for cryptocurrencies using the preprocessed dataset: `preprocessed_crypto_data.csv`.

The model uses **XGBoost**, trained on features including:
- Lagged prices
- Technical indicators (RSI, Bollinger Bands)
- Volatility measures
- Cyclical encodings for month and weekday

---

## 🚀 Features

- ✅ **Multi-Coin Predictions**  
  Trains a separate XGBoost model for each cryptocurrency and saves:
  - Evaluation metrics (MAE, RMSE, MAPE, directional accuracy)
  - Daily predicted prices

- 🛠️ **Feature Engineering**  
  - Lagged prices (1-day, 2-day, 3-day lags)
  - Rolling volatility
  - RSI (Relative Strength Index)
  - Bollinger Bands
  - Cyclical encodings (month, weekday)

- 🔍 **Interactive Prediction**  
  Users can input a coin name (e.g., `"Bitcoin"` for `coin_Bitcoin`) to:
  - View the predicted closing price
  - See evaluation metrics and feature importance
  - Plot historical vs. predicted prices using Matplotlib

- 📅 **Time-Series Cross-Validation**  
  Implements **5-fold time-based splitting**, preserving the temporal order for proper evaluation.

- 💾 **Output Files**
  - `coin_predictions.csv`: Contains predictions per coin
  - `coin_prediction_metrics.csv`: Contains model evaluation metrics

---

## 📦 Requirements

- Python 3.8 or higher

Install required libraries:

```bash
pip install pandas numpy scikit-learn xgboost ta matplotlib
```

---

## 📈 Example Output

Coming soon...

---

## 👤 Author

- **Hasan Ali**
- Dataset: [Kaggle Profile](https://www.kaggle.com/malikhasanali)

---

## 📃 License

This project is licensed under the MIT License.
