import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, GRU, Dense,
                                     Multiply, Softmax, Lambda)
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

# GPU 결정성 확보
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# ── 데이터 준비 (13_attention_lstm.py 동일) ────────────────────────
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

# Zero Change 기준선
zc_vr = np.sqrt(mean_squared_error(Y_val_w,  np.zeros(len(Y_val_w))))
zc_vm = mean_absolute_error(Y_val_w,  np.zeros(len(Y_val_w)))
zc_tr = np.sqrt(mean_squared_error(Y_test_w, np.zeros(len(Y_test_w))))
zc_tm = mean_absolute_error(Y_test_w, np.zeros(len(Y_test_w)))

print(f"Zero Change 기준 — Val RMSE:{zc_vr:.2f} MAE:{zc_vm:.2f} / Test RMSE:{zc_tr:.2f} MAE:{zc_tm:.2f}")
print("="*75)

# ── 모델 빌더 ──────────────────────────────────────────────────────
input_shape = (WINDOW, X_train.shape[1])

def build_lstm(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    inp = Input(shape=input_shape)
    x   = LSTM(32, return_sequences=False)(inp)
    out = Dense(1)(x)
    return Model(inp, out)

def build_attention_lstm(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    inp      = Input(shape=input_shape)
    lstm_out = LSTM(32, return_sequences=True)(inp)
    score    = Dense(1, activation='tanh')(lstm_out)
    weight   = Softmax(axis=1)(score)
    context  = Multiply()([lstm_out, weight])
    context  = Lambda(lambda x: K.sum(x, axis=1))(context)
    out      = Dense(1)(context)
    return Model(inp, out)

# ── 다중 시드 실험 ─────────────────────────────────────────────────
SEEDS = list(range(20))  # 0~19, 태현 선배님과 동일 조건

print(f"\n{'시드':>5} | {'모델':<16} | {'Val RMSE':>9} {'Val MAE':>8} {'Test RMSE':>10} {'Test MAE':>9} | gate")
print("-"*75)

summary = {"LSTM": {"pass": 0, "fail": 0}, "Attention-LSTM": {"pass": 0, "fail": 0}}

for seed in SEEDS:
    for name, builder in [("LSTM", build_lstm), ("Attention-LSTM", build_attention_lstm)]:
        tf.random.set_seed(seed)
        np.random.seed(seed)

        model = builder(seed)
        model.compile(optimizer="adam", loss="mse")
        model.fit(
            X_train_w, Y_train_w,
            validation_data=(X_val_w, Y_val_w),
            epochs=300, batch_size=32,
            callbacks=[EarlyStopping(monitor="val_loss", patience=20,
                                     restore_best_weights=True)],
            verbose=0,
        )

        vp = model.predict(X_val_w,  verbose=0).flatten()
        tp = model.predict(X_test_w, verbose=0).flatten()

        vr = np.sqrt(mean_squared_error(Y_val_w,  vp))
        vm = mean_absolute_error(Y_val_w,  vp)
        tr = np.sqrt(mean_squared_error(Y_test_w, tp))
        tm = mean_absolute_error(Y_test_w, tp)

        passed = (vr < zc_vr and vm < zc_vm and tr < zc_tr and tm < zc_tm)
        gate   = "✓" if passed else "✗"
        summary[name]["pass" if passed else "fail"] += 1

        print(f"{seed:>5} | {name:<16} | {vr:>8.2f}원 {vm:>7.2f}원 {tr:>9.2f}원 {tm:>8.2f}원 | {gate}")

# ── 최종 요약 ──────────────────────────────────────────────────────
print("="*75)
print(f"\n[검증 결과 요약] — {len(SEEDS)}개 시드 기준")
print(f"{'모델':<18} {'통과':>6} {'미통과':>7} {'통과율':>7}")
print("-"*40)
for name, r in summary.items():
    total    = r["pass"] + r["fail"]
    pass_pct = r["pass"] / total * 100
    print(f"{name:<18} {r['pass']:>6}회 {r['fail']:>6}회  {pass_pct:>6.0f}%")

print("\n결론:")
for name, r in summary.items():
    total    = r["pass"] + r["fail"]
    pass_pct = r["pass"] / total * 100
    if pass_pct == 100:
        print(f"  {name}: 모든 시드에서 통과 → 안정적")
    elif pass_pct >= 70:
        print(f"  {name}: 대부분 통과 ({pass_pct:.0f}%) → 비교적 안정적")
    elif pass_pct >= 40:
        print(f"  {name}: 절반 정도 통과 ({pass_pct:.0f}%) → 불안정, 시드 의존적")
    else:
        print(f"  {name}: 대부분 미통과 ({pass_pct:.0f}%) → 단일 시드 운이었을 가능성 높음")
