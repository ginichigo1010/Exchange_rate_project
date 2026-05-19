import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# ── 데이터 로드 ───────────────────────────────────────────────
df = pd.read_csv("raw_data_v2.csv", index_col="Date", parse_dates=True)

# ① 결측치 처리
df = df.ffill().dropna()
print(f"결측치 처리 후 shape: {df.shape}")

# ② 파생변수 추가
df["STOCK_DIFF"]     = df["KOSPI"] - df["SP500"]       # 한미 주식 수익률 차이
df["MA_DIFF"]        = df["MA5"]   - df["MA20"]         # 골든크로스 신호
df["JPY_KRW_SPREAD"] = df["JPY"]   - df["KRW"] / 10    # 엔-원 스프레드

df = df.dropna()
print(f"파생변수 추가 후 shape: {df.shape}")

# ③ 타겟 생성 (오늘 변수로 내일 환율 예측)
df["TARGET"] = df["KRW"].shift(-1)
df = df.dropna()

# ④ X / Y 분리
features = [
    "KRW", "DXY", "OIL", "KOSPI", "US10Y", "VIX", "SP500", "STOCK_DIFF",
    "JPY", "JPY_KRW_SPREAD",
    "MA5", "MA20", "RSI", "MA_DIFF",
    "SOX",
    "KR_RATE", "US_CPI",
]

X = df[features].values
Y = df["TARGET"].values

# ⑤ 분할 먼저 (시간 순서 유지, 8:1:1)
n = len(X)
train_end = int(n * 0.8)
val_end   = int(n * 0.9)

X_train_raw = X[:train_end]
X_val_raw   = X[train_end:val_end]
X_test_raw  = X[val_end:]
Y_train_raw = Y[:train_end]
Y_val_raw   = Y[train_end:val_end]
Y_test_raw  = Y[val_end:]

# ⑥ 정규화 — train 기준으로 fit, 나머지는 transform만
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_x.fit_transform(X_train_raw)
X_val   = scaler_x.transform(X_val_raw)
X_test  = scaler_x.transform(X_test_raw)

Y_train = scaler_y.fit_transform(Y_train_raw.reshape(-1, 1)).flatten()
Y_val   = scaler_y.transform(Y_val_raw.reshape(-1, 1)).flatten()
Y_test  = scaler_y.transform(Y_test_raw.reshape(-1, 1)).flatten()

print(f"\n훈련:  {len(X_train)}일")
print(f"검증:  {len(X_val)}일")
print(f"테스트: {len(X_test)}일")
print(f"총 입력 변수: {len(features)}개")

# ⑦ 저장
np.save("X_train_v2.npy", X_train)
np.save("X_val_v2.npy",   X_val)
np.save("X_test_v2.npy",  X_test)
np.save("Y_train_v2.npy", Y_train)
np.save("Y_val_v2.npy",   Y_val)
np.save("Y_test_v2.npy",  Y_test)
joblib.dump(scaler_x, "scaler_x_v2.pkl")
joblib.dump(scaler_y, "scaler_y_v2.pkl")
joblib.dump(features, "features_v2.pkl")

print("\n전처리 완료!")
