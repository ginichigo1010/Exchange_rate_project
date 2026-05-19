import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime

START = "2015-01-01"
END   = "2025-12-31"

# ① yfinance 7개
tickers = {
    "KRW"  : "KRW=X",
    "DXY"  : "DX-Y.NYB",
    "OIL"  : "CL=F",
    "KOSPI": "^KS11",
    "US10Y": "^TNX",
    "SP500": "^GSPC",
    "VIX"  : "^VIX",
}

df = pd.DataFrame()
for name, ticker in tickers.items():
    data = yf.download(ticker, start=START, end=END, auto_adjust=True)["Close"]
    df[name] = data

# ② FRED 2개 (월별 → 일별 변환)
kor_rate = web.DataReader("INTDSRKRM193N", "fred", START, END)  # 한국 기준금리
us_cpi   = web.DataReader("CPIAUCSL",      "fred", START, END)  # 미국 CPI

kor_rate = kor_rate.resample("D").ffill()
us_cpi   = us_cpi.resample("D").ffill()

df["KOR_RATE"] = kor_rate
df["US_CPI"]   = us_cpi

# 저장
df.to_csv("raw_data.csv")
print(f"수집 완료! shape: {df.shape}")
print(df.tail())
