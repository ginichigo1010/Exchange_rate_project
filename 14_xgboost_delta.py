import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, GRU, Dense, Dropout,
                                     Multiply, Softmax, Lambda)
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import xgboost as xgb

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 로드 & delta 타깃 생성 ─────────────────────────────────
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
df["TARGET"] = df["KRW"].shift(-1) - df["KRW"]
df = df.dropna()

features = [
    "KRW", "DXY", "OIL", "KOSPI", "US10Y", "VIX", "SP500", "STOCK_DIFF",
    "JPY", "JPY_KRW_SPREAD", "MA5", "MA20", "RSI", "MA_DIFF",
    "SOX", "KR_RATE", "US_CPI",
    "KRW_LAG1", "KRW_LAG5", "KRW_LAG10", "KRW_LAG20",
]
features_seq = features[:17]   # lag 없이 시퀀스 모델용 (기존과 동일)

X     = df[features].values
X_seq = df[features_seq].values
Y     = df["TARGET"].values

n         = len(X)
train_end = int(n * 0.8)
val_end   = int(n * 0.9)

X_train_raw = X[:train_end];        Y_train = Y[:train_end]
X_val_raw   = X[train_end:val_end]; Y_val   = Y[train_end:val_end]
X_test_raw  = X[val_end:];          Y_test  = Y[val_end:]

X_seq_train = X_seq[:train_end]
X_seq_val   = X_seq[train_end:val_end]
X_seq_test  = X_seq[val_end:]

# XGBoost는 정규화 없이 원본 사용 (트리 기반은 scale 불필요)
# 시퀀스 모델용 정규화
scaler_x = MinMaxScaler()
X_seq_train_sc = scaler_x.fit_transform(X_seq_train)
X_seq_val_sc   = scaler_x.transform(X_seq_val)
X_seq_test_sc  = scaler_x.transform(X_seq_test)

# ── XGBoost ───────────────────────────────────────────────────────
print("XGBoost 학습 중...")

# val 기준으로 n_estimators 탐색
best_rmse   = float("inf")
best_params = {}

for n_est in [100, 200, 300, 500]:
    for lr in [0.01, 0.05, 0.1]:
        for max_depth in [3, 5]:
            m = xgb.XGBRegressor(
                n_estimators=n_est,
                learning_rate=lr,
                max_depth=max_depth,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
            )
            m.fit(X_train_raw, Y_train)
            vp   = m.predict(X_val_raw)
            rmse = np.sqrt(mean_squared_error(Y_val, vp))
            if rmse < best_rmse:
                best_rmse   = rmse
                best_params = dict(n_estimators=n_est, learning_rate=lr, max_depth=max_depth)

print(f"XGBoost 최적 파라미터: {best_params}  (val RMSE: {best_rmse:.2f}원)")

xgb_model = xgb.XGBRegressor(**best_params, subsample=0.8,
                              colsample_bytree=0.8, random_state=42, verbosity=0)
xgb_model.fit(X_train_raw, Y_train)
xgb_val  = xgb_model.predict(X_val_raw)
xgb_test = xgb_model.predict(X_test_raw)

# ── 시퀀스 모델 (비교용) ──────────────────────────────────────────
WINDOW = 10

def make_window(X, Y, window):
    Xw, Yw = [], []
    for i in range(window, len(X)):
        Xw.append(X[i-window:i])
        Yw.append(Y[i])
    return np.array(Xw), np.array(Yw)

X_train_w, Y_train_w = make_window(X_seq_train_sc, Y_train, WINDOW)
X_val_w,   Y_val_w   = make_window(X_seq_val_sc,   Y_val,   WINDOW)
X_test_w,  Y_test_w  = make_window(X_seq_test_sc,  Y_test,  WINDOW)

def build_attention_lstm(input_shape):
    inp      = Input(shape=input_shape)
    lstm_out = LSTM(32, return_sequences=True)(inp)
    score    = Dense(1, activation='tanh')(lstm_out)
    weight   = Softmax(axis=1)(score)
    context  = Multiply()([lstm_out, weight])
    context  = Lambda(lambda x: K.sum(x, axis=1))(context)
    out      = Dense(1)(context)
    return Model(inp, out, name="Attention_LSTM")

input_shape = (WINDOW, X_seq_train_sc.shape[1])

attn_lstm = build_attention_lstm(input_shape)
attn_lstm.compile(optimizer="adam", loss="mse")
print("Attention-LSTM 학습 중...")
hist = attn_lstm.fit(
    X_train_w, Y_train_w,
    validation_data=(X_val_w, Y_val_w),
    epochs=300, batch_size=32,
    callbacks=[EarlyStopping(monitor="val_loss", patience=20,
                             restore_best_weights=True)],
    verbose=0,
)
print(f"Attention-LSTM 최적 epoch: {np.argmin(hist.history['val_loss']) + 1}")
attn_val  = attn_lstm.predict(X_val_w,  verbose=0).flatten()
attn_test = attn_lstm.predict(X_test_w, verbose=0).flatten()

# ── 평가 ──────────────────────────────────────────────────────────
def metrics(pred, actual):
    return (
        np.sqrt(mean_squared_error(actual, pred)),
        mean_absolute_error(actual, pred),
        r2_score(actual, pred),
    )

def trim(arr): return arr[WINDOW:]

# XGBoost는 window 없으므로 Y_val/Y_test 그대로, 시퀀스 모델은 Y_val_w/Y_test_w
zc_vr, zc_vm, _ = metrics(np.zeros(len(Y_val_w)), Y_val_w)
zc_tr, zc_tm, _ = metrics(np.zeros(len(Y_test_w)), Y_test_w)

# XGBoost val/test도 window만큼 앞부분 자르기 (공정 비교)
models = [
    ("Zero Change",    np.zeros(len(Y_val_w)),  np.zeros(len(Y_test_w))),
    ("XGBoost",        trim(xgb_val),            trim(xgb_test)),
    ("Attention-LSTM", attn_val,                 attn_test),
]

print("\n" + "="*75)
print(f"{'모델':<18} {'Val RMSE':>9} {'Val MAE':>9} {'Test RMSE':>10} {'Test MAE':>9}  gate")
print("-"*75)
rows = []
for name, vp, tp in models:
    vr, vm, _ = metrics(vp, Y_val_w)
    tr, tm, _ = metrics(tp, Y_test_w)
    passed = (vr < zc_vr and vm < zc_vm and tr < zc_tr and tm < zc_tm)
    gate = "✓ 통과" if passed else "✗ 미통과"
    print(f"{name:<18} {vr:>8.2f}원 {vm:>8.2f}원 {tr:>9.2f}원 {tm:>8.2f}원  {gate}")
    rows.append((name, vr, vm, tr, tm, passed))
print("="*75)
print("gate 기준: Val·Test RMSE·MAE 모두 Zero Change보다 낮아야 통과\n")

# ── 피처 중요도 출력 ───────────────────────────────────────────────
importance = xgb_model.feature_importances_
sorted_idx = np.argsort(importance)[::-1]
print("[XGBoost 피처 중요도 상위 10개]")
for i in sorted_idx[:10]:
    print(f"  {features[i]:<20s}: {importance[i]:.4f}")

# ── 시각화 ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) Val vs Test RMSE
x     = np.arange(len(rows))
width = 0.35
axes[0].bar(x - width/2, [r[1] for r in rows], width,
            label="Val RMSE",  color="steelblue", alpha=0.8)
axes[0].bar(x + width/2, [r[3] for r in rows], width,
            label="Test RMSE", color="tomato",    alpha=0.8)
axes[0].axhline(zc_vr, color="steelblue", linestyle="--", linewidth=1.0, alpha=0.6)
axes[0].axhline(zc_tr, color="tomato",    linestyle="--", linewidth=1.0, alpha=0.6,
                label=f"Zero Change (val {zc_vr:.2f} / test {zc_tr:.2f})")
axes[0].set_title("Val vs Test RMSE — XGBoost vs Attention-LSTM")
axes[0].set_ylabel("RMSE (원)")
axes[0].set_xticks(x)
axes[0].set_xticklabels([r[0] for r in rows], rotation=15)
axes[0].legend()

# (b) 테스트 구간 실제 vs 예측
axes[1].plot(Y_test_w,   label="실제 변화량",    color="steelblue",      linewidth=1.2)
axes[1].plot(trim(xgb_test), label="XGBoost",   color="darkorange",     linestyle="--", linewidth=1.0)
axes[1].plot(attn_test,  label="Attention-LSTM", color="mediumpurple",   linestyle="--", linewidth=1.0)
axes[1].axhline(0, color="lightcoral", linestyle="-", linewidth=1.0, label="Zero Change (0)")
axes[1].set_title("테스트 구간 실제 변화량 vs 예측")
axes[1].set_xlabel("테스트 시점")
axes[1].set_ylabel("변화량 (원)")
axes[1].legend()

plt.tight_layout()
plt.savefig("xgboost_compare.png", dpi=150)
plt.show()
print("저장 완료: xgboost_compare.png")
