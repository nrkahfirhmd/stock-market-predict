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
        st.error("Market Is Not Found! Try Something Else")
        return None

    data = hist.reset_index()
    
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
    
    # model = lr.fit(X_train, y_train)
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


def graphic(df, predictions):
    last = df["Close"].iloc[-23:].values
    pred = np.array(predictions)

    data = {
        "Forecast": np.concatenate((last, pred)),
        "Actual": np.concatenate((last, [None]*len(pred)))
    }
    plot_df = pd.DataFrame(data)

    st.line_chart(plot_df, color=["#6EDB8F", "#DBD035"])

st.title(":green[Stock Market] Prediction", anchor=False)
stock = st.text_input("Search Market", placeholder="Stock Market")

if stock:
    df = data_scraping(stock)
    
    if df is not None:
        current = df["Close"].iloc[-1]
        latest_date = df['Date'].iloc[-1]
        model, df, mae, rsme = train_model(df)
        result = predict(model, df)
        
        st.subheader(":green[Current Price] : " + str(round(current)))
        st.subheader(":green[30-Days Graphic]")
        graphic(df, result)
        st.header(":green[7-Days] Forecast")
        
        now = current
        i = 1
        for days in result:
            latest_date = latest_date + pd.Timedelta(days=1)
            with st.container():
                col1, col2 = st.columns(2)
                off = days - now

                if off < 0:
                    off = off * -1
                    percentage = off / now
                    change = ":red[(-" + str(round(percentage, 5)) + "%)]"
                else:
                    percentage = off / now
                    change = ":green[(+" + str(round(percentage, 5)) + "%)]"
                
                with col1:
                    st.subheader("Day-" + str(i) + " (" + str(pd.to_datetime(latest_date).strftime('%Y-%m-%d')) + ")")
                
                with col2:
                    st.subheader(str(round(days, 3)) + " " + change)
            
            now = days
            i = i + 1
        
        st.subheader(":red[BEWARE : ]")
        st.markdown("#### Average predictions are off by " + str(round(mae)) + " units from the actual values.")