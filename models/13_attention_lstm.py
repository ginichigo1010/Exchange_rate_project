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

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 로드 & delta 타깃 생성 ─────────────────────────────────
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

X_train_raw = X[:train_end];        Y_train = Y[:train_end]
X_val_raw   = X[train_end:val_end]; Y_val   = Y[train_end:val_end]
X_test_raw  = X[val_end:];          Y_test  = Y[val_end:]

scaler_x = MinMaxScaler()
X_train  = scaler_x.fit_transform(X_train_raw)
X_val    = scaler_x.transform(X_val_raw)
X_test   = scaler_x.transform(X_test_raw)

# ── window 시퀀스 생성 ────────────────────────────────────────────
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

def trim(arr): return arr[WINDOW:]

# ── 모델 빌더 ─────────────────────────────────────────────────────
def build_lstm(input_shape):
    inp = Input(shape=input_shape)
    x   = LSTM(32, return_sequences=False)(inp)
    x   = Dropout(0.0)(x)
    out = Dense(1)(x)
    return Model(inp, out, name="LSTM")

def build_gru(input_shape):
    inp = Input(shape=input_shape)
    x   = GRU(32, return_sequences=False)(inp)
    x   = Dropout(0.0)(x)
    out = Dense(1)(x)
    return Model(inp, out, name="GRU")

def build_attention_lstm(input_shape):
    """
    Attention-LSTM (ALFA 구조):
    1. LSTM이 모든 시점의 hidden state 출력 (return_sequences=True)
    2. Attention score: 각 시점의 중요도 계산 (Dense → Softmax)
    3. 가중합: 중요한 시점에 더 큰 가중치 부여
    4. Dense → 예측
    """
    inp      = Input(shape=input_shape)
    lstm_out = LSTM(32, return_sequences=True)(inp)   # (batch, window, 32)

    # Attention score
    score    = Dense(1, activation='tanh')(lstm_out)  # (batch, window, 1)
    weight   = Softmax(axis=1)(score)                 # (batch, window, 1) — 시점별 가중치

    # 가중합 (context vector)
    context  = Multiply()([lstm_out, weight])         # (batch, window, 32)
    context  = Lambda(lambda x: K.sum(x, axis=1))(context)  # (batch, 32)

    out = Dense(1)(context)
    return Model(inp, out, name="Attention_LSTM")

input_shape = (WINDOW, X_train.shape[1])
early_stop  = EarlyStopping(monitor="val_loss", patience=20,
                            restore_best_weights=True)

results = []
preds   = {}

for name, model in [("LSTM",           build_lstm(input_shape)),
                    ("GRU",            build_gru(input_shape)),
                    ("Attention-LSTM", build_attention_lstm(input_shape))]:
    model.compile(optimizer="adam", loss="mse")
    print(f"{name} 학습 중...")
    hist = model.fit(
        X_train_w, Y_train_w,
        validation_data=(X_val_w, Y_val_w),
        epochs=300, batch_size=32,
        callbacks=[EarlyStopping(monitor="val_loss", patience=20,
                                 restore_best_weights=True)],
        verbose=0,
    )
    best_ep = np.argmin(hist.history["val_loss"]) + 1
    print(f"  최적 epoch: {best_ep}")

    vp = model.predict(X_val_w,  verbose=0).flatten()
    tp = model.predict(X_test_w, verbose=0).flatten()
    preds[name] = (vp, tp)

# ── baseline ──────────────────────────────────────────────────────
zc_val  = np.zeros(len(Y_val_w))
zc_test = np.zeros(len(Y_test_w))

def metrics(pred, actual):
    return (
        np.sqrt(mean_squared_error(actual, pred)),
        mean_absolute_error(actual, pred),
        r2_score(actual, pred),
    )

zc_vr, zc_vm, _ = metrics(zc_val,  Y_val_w)
zc_tr, zc_tm, _ = metrics(zc_test, Y_test_w)

all_models = [("Zero Change", zc_val, zc_test)] + \
             [(n, vp, tp) for n, (vp, tp) in preds.items()]

print("\n" + "="*75)
print(f"{'모델':<18} {'Val RMSE':>9} {'Val MAE':>9} {'Test RMSE':>10} {'Test MAE':>9}  gate")
print("-"*75)
rows = []
for name, vp, tp in all_models:
    vr, vm, _ = metrics(vp, Y_val_w)
    tr, tm, _ = metrics(tp, Y_test_w)
    passed = (vr < zc_vr and vm < zc_vm and tr < zc_tr and tm < zc_tm)
    gate = "✓ 통과" if passed else "✗ 미통과"
    print(f"{name:<18} {vr:>8.2f}원 {vm:>8.2f}원 {tr:>9.2f}원 {tm:>8.2f}원  {gate}")
    rows.append((name, vr, vm, tr, tm, passed))
print("="*75)
print("gate 기준: Val·Test RMSE·MAE 모두 Zero Change보다 낮아야 통과\n")

# ── 시각화 ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) Val vs Test RMSE 비교
x     = np.arange(len(rows))
width = 0.35
val_r  = [r[1] for r in rows]
test_r = [r[3] for r in rows]
names  = [r[0] for r in rows]
bar_colors = ["lightcoral" if not r[5] and r[0] == "Zero Change"
              else ("mediumseagreen" if r[5] else "steelblue")
              for r in rows]

axes[0].bar(x - width/2, val_r,  width, label="Val RMSE",  color="steelblue", alpha=0.8)
axes[0].bar(x + width/2, test_r, width, label="Test RMSE", color="tomato",    alpha=0.8)
axes[0].axhline(zc_vr, color="steelblue", linestyle="--", linewidth=1.0, alpha=0.6)
axes[0].axhline(zc_tr, color="tomato",    linestyle="--", linewidth=1.0, alpha=0.6,
                label=f"Zero Change (val {zc_vr:.2f} / test {zc_tr:.2f})")
axes[0].set_title("Val vs Test RMSE — LSTM vs GRU vs Attention-LSTM")
axes[0].set_ylabel("RMSE (원)")
axes[0].set_xticks(x)
axes[0].set_xticklabels(names, rotation=15)
axes[0].legend()

# (b) 테스트 구간 실제 vs 예측
axes[1].plot(Y_test_w, label="실제 변화량", color="steelblue", linewidth=1.2)
colors_line = ["darkorange", "mediumseagreen", "mediumpurple"]
for (name, (vp, tp)), c in zip(preds.items(), colors_line):
    axes[1].plot(tp, label=name, color=c, linestyle="--", linewidth=1.0)
axes[1].axhline(0, color="lightcoral", linestyle="-", linewidth=1.0, label="Zero Change (0)")
axes[1].set_title("테스트 구간 실제 변화량 vs 예측")
axes[1].set_xlabel("테스트 시점")
axes[1].set_ylabel("변화량 (원)")
axes[1].legend()

plt.tight_layout()
plt.savefig("attention_lstm_compare.png", dpi=150)
plt.show()
print("저장 완료: attention_lstm_compare.png")
