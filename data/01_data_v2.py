import yfinance as yf
import pandas as pd
import requests

def calc_rsi(series, length=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(length).mean()
    loss  = (-delta.clip(upper=0)).rolling(length).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

START = "2015-01-01"
END   = "2025-12-31"

def get_fred(series_id):
    url = (f"https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}"
           f"&observation_start={START}"
           f"&observation_end={END}"
           f"&api_key=0c5adcbc3dee73468326f675d7a4b141"
           f"&file_type=json")
    r = requests.get(url, timeout=10)
    data = r.json()["observations"]
    df = pd.DataFrame(data)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df

# ── yfinance 수집 ─────────────────────────────────────────────
tickers = {
    "KRW"  : "KRW=X",
    "DXY"  : "DX-Y.NYB",
    "OIL"  : "CL=F",
    "KOSPI": "^KS11",
    "US10Y": "^TNX",
    "SP500": "^GSPC",
    "VIX"  : "^VIX",
    "JPY"  : "JPY=X",
    "SOX"  : "^SOX",
}

df = pd.DataFrame()
for name, ticker in tickers.items():
    data = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)["Close"]
    df[name] = data

# ── FRED 수집 (월별 → 일별) ───────────────────────────────────
print("한국 기준금리 수집 중...")
kr_rate = get_fred("INTDSRKRM193N").resample("D").ffill()
kr_rate.columns = ["KR_RATE"]

print("미국 CPI 수집 중...")
us_cpi = get_fred("CPIAUCSL").resample("D").ffill()
us_cpi.columns = ["US_CPI"]

df["KR_RATE"] = kr_rate
df["US_CPI"]  = us_cpi

# ── 결측치 처리 ───────────────────────────────────────────────
df = df.ffill()
df = df[df["KRW"].notna()].dropna()

# ── 파생 지표 (MA, RSI) ───────────────────────────────────────
df["MA5"]  = df["KRW"].rolling(5).mean()
df["MA20"] = df["KRW"].rolling(20).mean()
df["RSI"]  = calc_rsi(df["KRW"], length=14)

df = df.dropna()
df.index.name = "Date"

# 저장
df.to_csv("raw_data_v2.csv")
print(f"\n수집 완료! shape: {df.shape}")
print(f"변수 목록: {list(df.columns)}")
print(df.tail())
