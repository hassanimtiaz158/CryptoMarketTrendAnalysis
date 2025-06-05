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
