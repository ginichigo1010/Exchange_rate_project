import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

HORIZON = 5   # 예측 대상: 5일 후 변화량

# ── 데이터 로드 ────────────────────────────────────────────────────
df = pd.read_csv("raw_data_v2.csv", index_col="Date", parse_dates=True)
df = df.ffill().dropna()

df["STOCK_DIFF"]     = df["KOSPI"] - df["SP500"]
df["MA_DIFF"]        = df["MA5"]   - df["MA20"]
df["JPY_KRW_SPREAD"] = df["JPY"]   - df["KRW"] / 10

# KRW lag 피처 추가 (교수님 XGBoost 중요도 상위)
df["KRW_LAG1"]  = df["KRW"].shift(1)
df["KRW_LAG5"]  = df["KRW"].shift(5)
df["KRW_LAG10"] = df["KRW"].shift(10)
df["KRW_LAG20"] = df["KRW"].shift(20)

df = df.dropna()

# 타깃: 5일 후 환율 - 오늘 환율
df["TARGET"] = df["KRW"].shift(-HORIZON) - df["KRW"]
df = df.dropna()

features = [
    "KRW", "DXY", "OIL", "KOSPI", "US10Y", "VIX", "SP500", "STOCK_DIFF",
    "JPY", "JPY_KRW_SPREAD", "MA5", "MA20", "RSI", "MA_DIFF",
    "SOX", "KR_RATE", "US_CPI",
    "KRW_LAG1", "KRW_LAG5", "KRW_LAG10", "KRW_LAG20",   # lag 피처
]

X = df[features].values
Y = df["TARGET"].values

# ── 분할 (8:1:1, 시간 순서 유지) ──────────────────────────────────
n         = len(X)
train_end = int(n * 0.8)
val_end   = int(n * 0.9)

X_train_raw = X[:train_end];        Y_train = Y[:train_end]
X_val_raw   = X[train_end:val_end]; Y_val   = Y[train_end:val_end]
X_test_raw  = X[val_end:];          Y_test  = Y[val_end:]

print(f"훈련: {len(X_train_raw)}일 / 검증: {len(X_val_raw)}일 / 테스트: {len(X_test_raw)}일")
print(f"훈련 구간 평균 5일 변화량: {Y_train.mean():+.4f}원")

scaler_x = MinMaxScaler()
X_train  = scaler_x.fit_transform(X_train_raw)
X_val    = scaler_x.transform(X_val_raw)
X_test   = scaler_x.transform(X_test_raw)

# ── baseline ──────────────────────────────────────────────────────

# 1. Zero Change: 5일간 변화 없음
zc_val  = np.zeros(len(Y_val))
zc_test = np.zeros(len(Y_test))

# 2. Train Mean: 훈련 구간 평균 5일 변화량
train_mean_delta = Y_train.mean()
tm_val  = np.full(len(Y_val),  train_mean_delta)
tm_test = np.full(len(Y_test), train_mean_delta)

# ── 선형 모델 ──────────────────────────────────────────────────────

# 선형회귀
lr = LinearRegression()
lr.fit(X_train, Y_train)
lr_val  = lr.predict(X_val)
lr_test = lr.predict(X_test)

# LASSO
alphas = np.logspace(-4, 1, 60)
lasso_val_rmses = []
for a in alphas:
    m = Lasso(alpha=a, max_iter=20000)
    m.fit(X_train, Y_train)
    lasso_val_rmses.append(np.sqrt(mean_squared_error(Y_val, m.predict(X_val))))
best_lasso_alpha = alphas[np.argmin(lasso_val_rmses)]
print(f"LASSO 최적 alpha: {best_lasso_alpha:.6f}")
lasso = Lasso(alpha=best_lasso_alpha, max_iter=20000)
lasso.fit(X_train, Y_train)
lasso_val  = lasso.predict(X_val)
lasso_test = lasso.predict(X_test)

# Ridge
ridge_alphas = np.logspace(-3, 3, 100)
ridge_val_rmses = []
for a in ridge_alphas:
    m = Ridge(alpha=a)
    m.fit(X_train, Y_train)
    ridge_val_rmses.append(np.sqrt(mean_squared_error(Y_val, m.predict(X_val))))
best_ridge_alpha = ridge_alphas[np.argmin(ridge_val_rmses)]
print(f"Ridge  최적 alpha: {best_ridge_alpha:.4f}")
ridge = Ridge(alpha=best_ridge_alpha)
ridge.fit(X_train, Y_train)
ridge_val  = ridge.predict(X_val)
ridge_test = ridge.predict(X_test)

# ── LSTM ──────────────────────────────────────────────────────────
WINDOW = 10

def make_window(X, Y, window):
    Xw, Yw = [], []
    for i in range(window, len(X)):
        Xw.append(X[i-window:i])
        Yw.append(Y[i])
    return np.array(Xw), np.array(Yw)

X_train_w, Y_train_w = make_window(X_train, Y_train, WINDOW)
X_val_w,   Y_val_w   = make_window(X_val,   Y_val,   WINDOW)
X_test_w,  Y_test_w  = make_window(X_test,  Y_test,  WINDOW)

lstm = Sequential([
    LSTM(32, return_sequences=False, input_shape=(WINDOW, X_train.shape[1])),
    Dropout(0.0),
    Dense(1),
])
lstm.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(monitor="val_loss", patience=20,
                           restore_best_weights=True)

print("LSTM 학습 중...")
history = lstm.fit(
    X_train_w, Y_train_w,
    validation_data=(X_val_w, Y_val_w),
    epochs=300,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0,
)
best_epoch = np.argmin(history.history["val_loss"]) + 1
print(f"최적 epoch: {best_epoch}")

lstm_val_pred  = lstm.predict(X_val_w,  verbose=0).flatten()
lstm_test_pred = lstm.predict(X_test_w, verbose=0).flatten()

# ── 평가 ──────────────────────────────────────────────────────────
def metrics(pred, actual):
    return (
        np.sqrt(mean_squared_error(actual, pred)),
        mean_absolute_error(actual, pred),
        r2_score(actual, pred),
    )

# LSTM window 만큼 앞부분 잘라서 맞추기
def trim(arr): return arr[WINDOW:]

zc_vr, zc_vm, _ = metrics(trim(zc_val),  Y_val_w)
zc_tr, zc_tm, _ = metrics(trim(zc_test), Y_test_w)

models = [
    ("Zero Change", trim(zc_val),  trim(zc_test)),
    ("Train Mean",  trim(tm_val),  trim(tm_test)),
    ("선형회귀",     trim(lr_val),  trim(lr_test)),
    ("LASSO",       trim(lasso_val), trim(lasso_test)),
    ("Ridge",       trim(ridge_val), trim(ridge_test)),
    ("LSTM",        lstm_val_pred, lstm_test_pred),
]

print("\n" + "="*75)
print(f"{'모델':<16} {'Val RMSE':>9} {'Val MAE':>9} {'Test RMSE':>10} {'Test MAE':>9}  gate")
print("-"*75)
rows = []
for name, val_pred, test_pred in models:
    vr, vm, _ = metrics(val_pred,  Y_val_w)
    tr, tm, _ = metrics(test_pred, Y_test_w)
    passed = (vr < zc_vr and vm < zc_vm and tr < zc_tr and tm < zc_tm)
    gate = "✓ 통과" if passed else "✗ 미통과"
    print(f"{name:<16} {vr:>8.2f}원 {vm:>8.2f}원 {tr:>9.2f}원 {tm:>8.2f}원  {gate}")
    rows.append((name, vr, vm, tr, tm, passed))
print("="*75)
print(f"gate 기준: Val·Test RMSE·MAE 모두 Zero Change보다 낮아야 통과\n")

# LASSO 피처 계수 요약
nonzero = [(features[i], lasso.coef_[i]) for i in range(len(features)) if lasso.coef_[i] != 0]
nonzero.sort(key=lambda x: abs(x[1]), reverse=True)
print(f"[LASSO 생존 변수: {len(nonzero)}개 / 전체 {len(features)}개]")
for fname, coef in nonzero:
    print(f"  {fname:<20s}: {coef:+.6f}")

# ── 시각화 ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) Val vs Test RMSE
x     = np.arange(len(rows))
width = 0.35
names  = [r[0] for r in rows]
val_r  = [r[1] for r in rows]
test_r = [r[3] for r in rows]

axes[0].bar(x - width/2, val_r,  width, label="Val RMSE",  color="steelblue", alpha=0.8)
axes[0].bar(x + width/2, test_r, width, label="Test RMSE", color="tomato",    alpha=0.8)
axes[0].axhline(zc_vr, color="steelblue", linestyle="--", linewidth=1.0, alpha=0.6)
axes[0].axhline(zc_tr, color="tomato",    linestyle="--", linewidth=1.0, alpha=0.6,
                label=f"Zero Change (val {zc_vr:.2f} / test {zc_tr:.2f})")
axes[0].set_title(f"Val vs Test RMSE 비교 — {HORIZON}일 변화량 예측")
axes[0].set_ylabel("RMSE (원)")
axes[0].set_xticks(x)
axes[0].set_xticklabels(names, rotation=15)
axes[0].legend()

# (b) 테스트 구간 실제 vs 예측
axes[1].plot(Y_test_w,       label="실제 5일 변화량", color="steelblue",      linewidth=1.2)
axes[1].plot(lstm_test_pred, label="LSTM",           color="darkorange",     linestyle="--", linewidth=1.0)
axes[1].plot(trim(lasso_test), label="LASSO",        color="mediumseagreen", linestyle="--", linewidth=1.0)
axes[1].plot(trim(ridge_test), label="Ridge",        color="mediumpurple",   linestyle="--", linewidth=1.0)
axes[1].axhline(0, color="lightcoral", linestyle="-", linewidth=1.0, label="Zero Change (0)")
axes[1].set_title(f"테스트 구간 실제 {HORIZON}일 변화량 vs 예측")
axes[1].set_xlabel("테스트 시점")
axes[1].set_ylabel("변화량 (원)")
axes[1].legend()

plt.tight_layout()
plt.savefig("5day_compare.png", dpi=150)
plt.show()
print("저장 완료: 5day_compare.png")
