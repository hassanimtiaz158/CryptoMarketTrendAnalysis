import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from final_code import process_coin_data, features

st.set_page_config(page_title="Crypto Price Predictor", layout="wide")
st.title("🪙 Cryptocurrency Price Predictor")
st.markdown("**Explore, Analyze, and Predict Cryptocurrency Prices using Machine Learning**")

@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_crypto_data1.csv")
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df = df.sort_values(['Coin', 'Date'])
    return df

df = load_data()
available_coins = sorted([c.replace("coin_", "") for c in df['Coin'].unique()])

# Sidebar
st.sidebar.header("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "EDA", "Model & Prediction", "Conclusion"])
selected_coin = st.sidebar.selectbox("📌 Select a Coin", available_coins)


def get_coin_df(coin_name):
    coin_df = df[df['Coin'] == f"coin_{coin_name}"].copy()
    return process_coin_data(coin_df)

coin_df = get_coin_df(selected_coin) if selected_coin else pd.DataFrame()

# Page 1: Introduction
if page == "Introduction":
    st.header("📌 Project Overview")
    st.markdown("""
    Welcome to the Cryptocurrency Price Predictor project! This tool is designed to:
    - Analyze cryptocurrency data with **interactive visualizations**
    - Train a **machine learning model** to predict next-day closing prices
    - Provide **real-time predictions** based on historical trends

    **Dataset Highlights**:
    - Coins: Bitcoin, Ethereum, Litecoin, etc.
    - Features: Prices, Volume, Technical Indicators (MA, RSI, Bollinger Bands)
    - Engineered Features: Lag values, seasonal encodings

    Let's get started! Select a tab from the sidebar to explore more.
    """)
    
elif page == "EDA":
    st.header(f"📊 Exploratory Data Analysis for {selected_coin}")
    
    if coin_df.empty:
        st.warning("Not enough data to analyze this coin.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Summary Statistics")
            st.write(coin_df.describe())

        with col2:
            st.subheader("Missing Value Count")
            st.write(coin_df.isnull().sum())

        st.subheader("📈 Close Price Trend")
        st.line_chart(coin_df.set_index("Date")["Close"])

        st.subheader("📦 Volume Distribution")
        fig1, ax1 = plt.subplots()
        sns.histplot(coin_df["Volume"], bins=30, kde=True, ax=ax1)
        ax1.set_xlabel("Volume")
        ax1.set_ylabel("Frequency")
        st.pyplot(fig1)

        st.subheader("🔗 Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.heatmap(coin_df[features].corr(), cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)


elif page == "Model & Prediction":
    st.header(f"📈 Model Training & Prediction for {selected_coin}")
    
    if len(coin_df) < 30:
        st.error("Not enough data to train a model for this coin.")
    else:
        X = coin_df[features]
        y = coin_df['Close']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_scaled, y)

        next_input = X_scaled[-1].reshape(1, -1)
        next_pred = model.predict(next_input)[0]
        last_close = y.iloc[-1]
        last_date = coin_df['Date'].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)

        st.success(f"✅ Last Close on **{last_date.date()}**: **${last_close:.2f}**")
        st.success(f"📊 Predicted Close for **{next_date.date()}**: **${next_pred:.2f}**")


        st.subheader("📊 Top 10 Important Features")
        importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(10)
        fig3, ax3 = plt.subplots()
        importances.plot(kind="barh", color="skyblue", ax=ax3)
        ax3.set_xlabel("Importance")
        ax3.set_title("Feature Importance")
        st.pyplot(fig3)


        st.subheader("📉 Historical vs Predicted Close")
        fig4, ax4 = plt.subplots()
        ax4.plot(coin_df['Date'], coin_df['Close'], label="Historical", color="blue")
        ax4.scatter([next_date], [next_pred], label="Predicted", color="red")
        ax4.set_title(f"{selected_coin} Close Price Prediction")
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Close Price (USD)")
        ax4.legend()
        st.pyplot(fig4)

# Page 4: Conclusion
elif page == "Conclusion":
    st.header("📘 Conclusion")
    st.markdown(f"""
    The project demonstrates how we can **forecast cryptocurrency prices** using machine learning and interactive tools.

    ### 🔍 Key Takeaways:
    - XGBoost was effective in modeling crypto price trends.
    - Lag features and technical indicators improved prediction accuracy.
    - Streamlit provides an intuitive way to visualize and interact with data.

    ### 🚀 Next Steps:
    - Integrate real-time data feeds using APIs
    - Try deep learning models like LSTM
    - Add functionality to predict for custom dates or ranges
    """)

    st.balloons()
