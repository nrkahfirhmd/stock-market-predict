import streamlit as st 
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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

def train_model(data):
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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    lr = LinearRegression()
    y_pred = lr.fit(X_train, y_train).predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rsme = np.sqrt(mean_squared_error(y_test, y_pred))
    
    model = lr.fit(X, y)
    
    return model, df, mae, rsme

def predict(model, data, days=7):
    lr = model
    df = data.copy()
    
    predictions = []
    latest_data = df.iloc[-1].copy()
    
    for day in range(days):
        latest = np.array([
            latest_data['Close_Lag1'],
            latest_data['Close_Lag2'],
            latest_data['Volume_Lag1'],
            latest_data['MA5'],
            latest_data['MA10']
        ]).reshape(1, -1)
        
        y_pred = lr.predict(latest)[0]
        predictions.append(y_pred)
        
        new_data = {
            'Close': y_pred,
            'Close_Lag1': y_pred,
            'Close_Lag2': latest_data['Close_Lag1'],
            'Volume_Lag1': latest_data['Volume_Lag1'], 
            'MA5': (df['Close'].iloc[-4:].sum() + y_pred) / 5 if len(df) >= 4 else np.nan,
            'MA10': (df['Close'].iloc[-9:].sum() + y_pred) / 10 if len(df) >= 9 else np.nan
        }
        
        latest_data = pd.Series(new_data)
        df = pd.concat([df, pd.DataFrame(new_data, index=[0])], ignore_index=True)

    return predictions

stock = st.text_input("Search Market", placeholder="Stock Market")

if stock:
    df = data_scraping(stock)
    
    if df is not None:
        model, df, mae, rsme = train_model(df)
        st.write(predict(model, df))
        st.write("with mae = " + str(mae) + " rsme = " + str(rsme))