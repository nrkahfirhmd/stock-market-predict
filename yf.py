import yfinance as yf
import pandas as pd

nvda = yf.Ticker("ASII.JK")

hist = nvda.history(period="max")

data = pd.DataFrame(hist)

data.to_csv('price.csv')