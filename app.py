import streamlit as st 
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

def data_scraping(stock):
    market = yf.Ticker(stock)

    hist = market.history(period="max")
    
    if hist.empty:
        st.write("GAADA KANG :(")
        return None

    data = pd.DataFrame(hist)
    
    return data

def model_train(data):
    df = data
    
    df = df.sort_values('Date')
    
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    df['Volume_Lag1'] = df['Volume'].shift(1)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    
    df = df.dropna()
    
    features = ['Close_Lag1', 'Close_Lag2', 'Volume_Lag1', 'MA5', 'MA10']
    
    X = df[features]
    y = df['Close']
    
    latest_data = df.iloc[-1]

    new = np.array([
        latest_data['Close_Lag1'],
        latest_data['Close_Lag2'],
        latest_data['Volume_Lag1'],
        latest_data['MA5'],
        latest_data['MA10']
    ]).reshape(1, -1)
    
    lr = LinearRegression()
    y_pred = lr.fit(X, y).predict(new)
    # mae = mean_absolute_error(y_test, y_pred)
    # rsme = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return y_pred

stock = st.text_input("Search Market", placeholder="Stock Market")

if stock:
    df = data_scraping(stock)
    
    if df is not None:
        st.write(model_train(df)[0])