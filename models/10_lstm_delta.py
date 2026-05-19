import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 로드 & delta 타깃 생성 (09_delta_compare.py 동일) ────────
df = pd.read_csv("raw_data_v2.csv", index_col="Date", parse_dates=True)
df = df.ffill().dropna()

df["STOCK_DIFF"]     = df["KOSPI"] - df["SP500"]
df["MA_DIFF"]        = df["MA5"]   - df["MA20"]
df["JPY_KRW_SPREAD"] = df["JPY"]   - df["KRW"] / 10
df = df.dropna()

df["TARGET"] = df["KRW"].shift(-1) - df["KRW"]
df = df.dropna()

features = [
    "KRW", "DXY", "OIL", "KOSPI", "US10Y", "VIX", "SP500", "STOCK_DIFF",
    "JPY", "JPY_KRW_SPREAD", "MA5", "MA20", "RSI", "MA_DIFF",
    "SOX", "KR_RATE", "US_CPI",
]

X = df[features].values
Y = df["TARGET"].values

n         = len(X)
train_end = int(n * 0.8)
val_end   = int(n * 0.9)

X_train_raw = X[:train_end];      Y_train = Y[:train_end]
X_val_raw   = X[train_end:val_end]; Y_val   = Y[train_end:val_end]
X_test_raw  = X[val_end:];        Y_test  = Y[val_end:]

scaler_x = MinMaxScaler()
X_train  = scaler_x.fit_transform(X_train_raw)
X_val    = scaler_x.transform(X_val_raw)
X_test   = scaler_x.transform(X_test_raw)

# ── baseline & 선형 모델 (비교용) ─────────────────────────────────
zc_val  = np.zeros(len(Y_val))
zc_test = np.zeros(len(Y_test))

alphas = np.logspace(-4, 0, 50)
lasso_val_rmses = []
for a in alphas:
    m = Lasso(alpha=a, max_iter=20000)
    m.fit(X_train, Y_train)
    lasso_val_rmses.append(np.sqrt(mean_squared_error(Y_val, m.predict(X_val))))
best_alpha = alphas[np.argmin(lasso_val_rmses)]
lasso = Lasso(alpha=best_alpha, max_iter=20000)
lasso.fit(X_train, Y_train)
lasso_val  = lasso.predict(X_val)
lasso_test = lasso.predict(X_test)

ridge_alphas = np.logspace(-3, 3, 100)
ridge_val_rmses = []
for a in ridge_alphas:
    m = Ridge(alpha=a)
    m.fit(X_train, Y_train)
    ridge_val_rmses.append(np.sqrt(mean_squared_error(Y_val, m.predict(X_val))))
best_ridge_alpha = ridge_alphas[np.argmin(ridge_val_rmses)]
ridge = Ridge(alpha=best_ridge_alpha)
ridge.fit(X_train, Y_train)
ridge_val  = ridge.predict(X_val)
ridge_test = ridge.predict(X_test)

# ── LSTM — window 시퀀스 생성 ─────────────────────────────────────
WINDOW = 10   # 교수님 최적값

def make_window(X, Y, window):
    Xw, Yw = [], []
    for i in range(window, len(X)):
        Xw.append(X[i-window:i])
        Yw.append(Y[i])
    return np.array(Xw), np.array(Yw)

X_train_w, Y_train_w = make_window(X_train, Y_train, WINDOW)
X_val_w,   Y_val_w   = make_window(X_val,   Y_val,   WINDOW)
X_test_w,  Y_test_w  = make_window(X_test,  Y_test,  WINDOW)

# ── LSTM 모델 학습 ────────────────────────────────────────────────
lstm = Sequential([
    LSTM(32, return_sequences=False, input_shape=(WINDOW, X_train.shape[1])),
    Dropout(0.0),
    Dense(1),
])
lstm.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(monitor="val_loss", patience=10,
                           restore_best_weights=True)

print("LSTM 학습 중...")
history = lstm.fit(
    X_train_w, Y_train_w,
    validation_data=(X_val_w, Y_val_w),
    epochs=150,
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

# LSTM은 window 만큼 앞부분이 잘리므로 맞춰서 자르기
zc_val_w    = zc_val[WINDOW:]
zc_test_w   = zc_test[WINDOW:]
lasso_val_w = lasso_val[WINDOW:]
lasso_tst_w = lasso_test[WINDOW:]
ridge_val_w = ridge_val[WINDOW:]
ridge_tst_w = ridge_test[WINDOW:]

zc_vr, zc_vm, _ = metrics(zc_val_w,  Y_val_w)
zc_tr, zc_tm, _ = metrics(zc_test_w, Y_test_w)

models = [
    ("Zero Change", zc_val_w,    zc_test_w),
    ("LASSO",       lasso_val_w, lasso_tst_w),
    ("Ridge",       ridge_val_w, ridge_tst_w),
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
print("gate 기준: Val·Test RMSE·MAE 모두 Zero Change보다 낮아야 통과\n")

# ── 시각화 ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) Val vs Test RMSE 비교
x      = np.arange(len(rows))
width  = 0.35
names  = [r[0] for r in rows]
val_r  = [r[1] for r in rows]
test_r = [r[3] for r in rows]

axes[0].bar(x - width/2, val_r,  width, label="Val RMSE",  color="steelblue",  alpha=0.8)
axes[0].bar(x + width/2, test_r, width, label="Test RMSE", color="tomato",     alpha=0.8)
axes[0].axhline(zc_vr, color="steelblue", linestyle="--", linewidth=1.0, alpha=0.6)
axes[0].axhline(zc_tr, color="tomato",    linestyle="--", linewidth=1.0, alpha=0.6,
                label=f"Zero Change (val {zc_vr:.2f} / test {zc_tr:.2f})")
axes[0].set_title("Val vs Test RMSE 비교 — delta 예측")
axes[0].set_ylabel("RMSE (원)")
axes[0].set_xticks(x)
axes[0].set_xticklabels(names)
axes[0].legend()

# (b) 테스트 구간 실제 delta vs 예측
axes[1].plot(Y_test_w,       label="실제 변화량",  color="steelblue",      linewidth=1.2)
axes[1].plot(lstm_test_pred, label="LSTM",        color="darkorange",     linestyle="--", linewidth=1.0)
axes[1].plot(lasso_tst_w,    label="LASSO",       color="mediumseagreen", linestyle="--", linewidth=1.0)
axes[1].axhline(0, color="lightcoral", linestyle="-", linewidth=1.0, label="Zero Change (0)")
axes[1].set_title("테스트 구간 실제 변화량 vs 예측")
axes[1].set_xlabel("테스트 시점")
axes[1].set_ylabel("변화량 (원)")
axes[1].legend()

plt.tight_layout()
plt.savefig("lstm_delta_compare.png", dpi=150)
plt.show()
print("저장 완료: lstm_delta_compare.png")
