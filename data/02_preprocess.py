import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# 데이터 불러오기
df = pd.read_csv("raw_data.csv", index_col="Date", parse_dates=True)

# ① 결측치 처리
df = df.ffill().dropna()
print(f"결측치 처리 후: {df.shape}")

# ② 타겟 생성 (오늘 변수로 내일 환율 예측)
df["TARGET"] = df["KRW"].shift(-1)
df = df.dropna()

# ③ X / Y 분리
feature_cols = ["KRW","DXY","OIL","KOSPI","US10Y","SP500","VIX","KOR_RATE","US_CPI"]
X = df[feature_cols].values
Y = df["TARGET"].values

# ④ 8:1:1 분할 (시간 순서 유지) — 정규화 전에 먼저 분할
n = len(X)
train_end = int(n * 0.8)
val_end   = int(n * 0.9)

X_train_raw = X[:train_end]
X_val_raw   = X[train_end:val_end]
X_test_raw  = X[val_end:]
Y_train_raw = Y[:train_end]
Y_val_raw   = Y[train_end:val_end]
Y_test_raw  = Y[val_end:]

# ⑤ 정규화 — train 기준으로 fit, 나머지는 transform만
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_x.fit_transform(X_train_raw)
X_val   = scaler_x.transform(X_val_raw)
X_test  = scaler_x.transform(X_test_raw)

Y_train = scaler_y.fit_transform(Y_train_raw.reshape(-1,1)).flatten()
Y_val   = scaler_y.transform(Y_val_raw.reshape(-1,1)).flatten()
Y_test  = scaler_y.transform(Y_test_raw.reshape(-1,1)).flatten()

print(f"훈련: {len(X_train)}일 / 검증: {len(X_val)}일 / 테스트: {len(X_test)}일")

# ⑥ 저장
np.save("X_train.npy", X_train)
np.save("X_val.npy",   X_val)
np.save("X_test.npy",  X_test)
np.save("Y_train.npy", Y_train)
np.save("Y_val.npy",   Y_val)
np.save("Y_test.npy",  Y_test)
joblib.dump(scaler_x, "scaler_x.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")

print("전처리 완료!")
