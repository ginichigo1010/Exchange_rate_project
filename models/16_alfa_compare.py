
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Layer, Multiply, Softmax, Lambda)
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 준비 ────────────────────────────────────────────────────
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

# ── ALFA Feature Attention 레이어 (논문 구조 그대로) ───────────────
class FeatureAttention(Layer):
    """
    ALFA 논문의 핵심: 피처 축으로 attention 계산
    - 시간 축이 아니라 '어느 변수가 중요한지'를 학습
    """
    def __init__(self, num_features, hidden_units, **kwargs):
        super(FeatureAttention, self).__init__(**kwargs)
        self.num_features = num_features
        self.hidden_units = hidden_units

    def build(self, input_shape):
        self.feature_attention  = Dense(self.num_features, activation="softmax",
                                        name="feature_attention")
        self.hidden_to_features = Dense(self.num_features, activation="linear",
                                        name="hidden_to_features")
        super(FeatureAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs: (batch, seq_len, hidden_units)
        attn_weights     = self.feature_attention(inputs)   # (batch, seq_len, num_features)
        projected        = self.hidden_to_features(inputs)  # (batch, seq_len, num_features)
        weighted         = attn_weights * projected          # (batch, seq_len, num_features)
        context_vector   = K.sum(weighted, axis=-1)         # (batch, seq_len)
        return context_vector, attn_weights

    def get_config(self):
        config = super().get_config()
        config.update({"num_features": self.num_features, "hidden_units": self.hidden_units})
        return config

# ── 모델 빌더 ──────────────────────────────────────────────────────
input_shape  = (WINDOW, X_train.shape[1])
num_features = X_train.shape[1]

def build_lstm(seed):
    tf.random.set_seed(seed); np.random.seed(seed)
    inp  = Input(shape=input_shape)
    x    = LSTM(32, return_sequences=False)(inp)
    out  = Dense(1)(x)
    return Model(inp, out, name="LSTM")

def build_temporal_attn_lstm(seed):
    """우리가 기존에 썼던 Temporal Attention (시간 축)"""
    tf.random.set_seed(seed); np.random.seed(seed)
    inp      = Input(shape=input_shape)
    lstm_out = LSTM(32, return_sequences=True)(inp)
    score    = Dense(1, activation='tanh')(lstm_out)
    weight   = Softmax(axis=1)(score)
    context  = Multiply()([lstm_out, weight])
    context  = Lambda(lambda x: K.sum(x, axis=1))(context)
    out      = Dense(1)(context)
    return Model(inp, out, name="Temporal_Attn_LSTM")

def build_alfa(seed):
    """ALFA 논문 구조: Feature Attention (피처 축)"""
    tf.random.set_seed(seed); np.random.seed(seed)
    inp         = Input(shape=input_shape)
    lstm_out    = LSTM(64, return_sequences=True, name="LSTM")(inp)
    attn_layer  = FeatureAttention(num_features=num_features, hidden_units=128,
                                   name="FeatureAttention")
    context, _  = attn_layer(lstm_out)     # (batch, seq_len)
    out         = Dense(1, activation="linear", name="Output")(context)
    model       = Model(inp, out, name="ALFA")
    adam = tf.keras.optimizers.Adam(
        clipvalue=0.5, learning_rate=0.001,
        beta_1=0.9, beta_2=0.99, epsilon=1e-7, decay=0.001
    )
    model.compile(loss="mean_squared_error", optimizer=adam)
    return model

# ── Zero Change 기준선 ─────────────────────────────────────────────
zc_vr = np.sqrt(mean_squared_error(Y_val_w,  np.zeros(len(Y_val_w))))
zc_vm = mean_absolute_error(Y_val_w,  np.zeros(len(Y_val_w)))
zc_tr = np.sqrt(mean_squared_error(Y_test_w, np.zeros(len(Y_test_w))))
zc_tm = mean_absolute_error(Y_test_w, np.zeros(len(Y_test_w)))

print(f"Zero Change — Val RMSE:{zc_vr:.4f} MAE:{zc_vm:.4f} / Test RMSE:{zc_tr:.4f} MAE:{zc_tm:.4f}")

# ── 다중 시드 실험 (시드 0~19) ────────────────────────────────────
SEEDS = list(range(20))

builders = [
    ("LSTM",              build_lstm),
    ("Temporal-Attn-LSTM", build_temporal_attn_lstm),
    ("ALFA",              build_alfa),
]

summary = {name: {"pass": 0, "fail": 0, "test_rmse": []} for name, _ in builders}

print(f"\n{'시드':>5} | {'모델':<22} | {'Val RMSE':>9} {'Val MAE':>8} {'Test RMSE':>10} {'Test MAE':>9} | gate")
print("-"*85)

for seed in SEEDS:
    for name, builder in builders:
        tf.random.set_seed(seed); np.random.seed(seed)

        model = builder(seed)
        if name != "ALFA":
            model.compile(optimizer="adam", loss="mse")

        patience = 40 if name == "ALFA" else 20
        model.fit(
            X_train_w, Y_train_w,
            validation_data=(X_val_w, Y_val_w),
            epochs=300, batch_size=32,
            callbacks=[EarlyStopping(monitor="val_loss", patience=patience,
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
        summary[name]["test_rmse"].append(tr)

        print(f"{seed:>5} | {name:<22} | {vr:>8.4f}원 {vm:>7.4f}원 {tr:>9.4f}원 {tm:>8.4f}원 | {gate}")

# ── 최종 요약 ──────────────────────────────────────────────────────
print("="*85)
print(f"\n[검증 결과 요약] — {len(SEEDS)}개 시드 기준")
print(f"{'모델':<24} {'통과':>6} {'미통과':>7} {'통과율':>7} {'평균 Test RMSE':>15} {'σ':>8}")
print("-"*70)
for name, r in summary.items():
    total    = r["pass"] + r["fail"]
    pass_pct = r["pass"] / total * 100
    mean_r   = np.mean(r["test_rmse"])
    std_r    = np.std(r["test_rmse"])
    print(f"{name:<24} {r['pass']:>6}회 {r['fail']:>6}회  {pass_pct:>6.0f}%  {mean_r:>13.4f}원  {std_r:>6.4f}원")

# ── 시각화 ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) 통과율 비교
names     = list(summary.keys())
pass_rates = [summary[n]["pass"] / len(SEEDS) * 100 for n in names]
colors    = ["steelblue", "mediumpurple", "tomato"]
axes[0].bar(names, pass_rates, color=colors, alpha=0.8)
axes[0].axhline(50, color="gray", linestyle="--", linewidth=1)
axes[0].set_title(f"gate 통과율 비교 ({len(SEEDS)}개 시드)")
axes[0].set_ylabel("통과율 (%)")
axes[0].set_ylim(0, 105)
for i, v in enumerate(pass_rates):
    axes[0].text(i, v + 1.5, f"{v:.0f}%", ha='center', fontsize=11)

# (b) Test RMSE 분포 (boxplot)
data = [summary[n]["test_rmse"] for n in names]
bp   = axes[1].boxplot(data, labels=names, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].axhline(zc_tr, color="red", linestyle="--", linewidth=1.2,
                label=f"Zero Change ({zc_tr:.4f}원)")
axes[1].set_title("Test RMSE 분포 비교")
axes[1].set_ylabel("Test RMSE (원)")
axes[1].legend()

plt.tight_layout()
plt.savefig("alfa_compare.png", dpi=150)
print("\n저장 완료: alfa_compare.png")
