import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

scaler_y = joblib.load("scaler_y.pkl")

X_train = np.load("X_train.npy")
X_val   = np.load("X_val.npy")
X_test  = np.load("X_test.npy")
Y_train = np.load("Y_train.npy")
Y_val   = np.load("Y_val.npy")
Y_test  = np.load("Y_test.npy")

# ── 선형회귀 ──────────────────────────────────────
lr = LinearRegression()
lr.fit(X_train, Y_train)
lr_pred = lr.predict(X_test)

Y_real    = scaler_y.inverse_transform(Y_test.reshape(-1,1)).flatten()
lr_real   = scaler_y.inverse_transform(lr_pred.reshape(-1,1)).flatten()

lr_rmse = np.sqrt(mean_squared_error(Y_real, lr_real))
lr_mae  = mean_absolute_error(Y_real, lr_real)
lr_r2   = r2_score(Y_real, lr_real)

# ── LASSO ────────────────────────────────────────
alphas = np.logspace(-4, 0, 50)
val_rmses = []
for alpha in alphas:
    m = Lasso(alpha=alpha, max_iter=10000)
    m.fit(X_train, Y_train)
    pv = m.predict(X_val)
    Y_val_real = scaler_y.inverse_transform(Y_val.reshape(-1,1)).flatten()
    pv_real    = scaler_y.inverse_transform(pv.reshape(-1,1)).flatten()
    val_rmses.append(np.sqrt(mean_squared_error(Y_val_real, pv_real)))

best_alpha = alphas[np.argmin(val_rmses)]
lasso = Lasso(alpha=best_alpha, max_iter=10000)
lasso.fit(X_train, Y_train)
lasso_pred = lasso.predict(X_test)
lasso_real = scaler_y.inverse_transform(lasso_pred.reshape(-1,1)).flatten()

lasso_rmse = np.sqrt(mean_squared_error(Y_real, lasso_real))
lasso_mae  = mean_absolute_error(Y_real, lasso_real)
lasso_r2   = r2_score(Y_real, lasso_real)

# ── LSTM ─────────────────────────────────────────
WINDOW = 30

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
    LSTM(64, return_sequences=True, input_shape=(WINDOW, X_train.shape[1])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
lstm.compile(optimizer="adam", loss="mse")
lstm.fit(X_train_w, Y_train_w,
         validation_data=(X_val_w, Y_val_w),
         epochs=50, batch_size=32, verbose=0)

lstm_pred = lstm.predict(X_test_w)
Y_real_w      = scaler_y.inverse_transform(Y_test_w.reshape(-1,1)).flatten()
lstm_real     = scaler_y.inverse_transform(lstm_pred).flatten()

lstm_rmse = np.sqrt(mean_squared_error(Y_real_w, lstm_real))
lstm_mae  = mean_absolute_error(Y_real_w, lstm_real)
lstm_r2   = r2_score(Y_real_w, lstm_real)

# ── 성능 비교표 출력 ──────────────────────────────
print("\n" + "="*45)
print(f"{'모델':<12} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("-"*45)
print(f"{'선형회귀':<12} {lr_rmse:>7.2f}원 {lr_mae:>7.2f}원 {lr_r2:>8.4f}")
print(f"{'LASSO':<12} {lasso_rmse:>7.2f}원 {lasso_mae:>7.2f}원 {lasso_r2:>8.4f}")
print(f"{'LSTM':<12} {lstm_rmse:>7.2f}원 {lstm_mae:>7.2f}원 {lstm_r2:>8.4f}")
print("="*45)

# ── 비교 시각화 ───────────────────────────────────
# LSTM 결과는 window 크기만큼 앞이 잘리므로 선형회귀도 같은 구간으로 맞춤
offset = len(Y_real) - len(Y_real_w)
lr_real_aligned    = lr_real[offset:]
lasso_real_aligned = lasso_real[offset:]

fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)

axes[0].plot(Y_real_w,        label="실제 환율",   color="steelblue")
axes[0].plot(lr_real_aligned, label="선형회귀 예측", color="tomato", linestyle="--")
axes[0].set_title(f"선형회귀 — RMSE={lr_rmse:.2f}원  R²={lr_r2:.4f}")
axes[0].legend()

axes[1].plot(Y_real_w,           label="실제 환율",  color="steelblue")
axes[1].plot(lasso_real_aligned, label="LASSO 예측", color="mediumseagreen", linestyle="--")
axes[1].set_title(f"LASSO (α={best_alpha:.5f}) — RMSE={lasso_rmse:.2f}원  R²={lasso_r2:.4f}")
axes[1].legend()

axes[2].plot(Y_real_w,  label="실제 환율",  color="steelblue")
axes[2].plot(lstm_real, label="LSTM 예측", color="darkorange", linestyle="--")
axes[2].set_title(f"LSTM — RMSE={lstm_rmse:.2f}원  R²={lstm_r2:.4f}")
axes[2].legend()

plt.tight_layout()
plt.savefig("compare_result.png")
plt.show()
