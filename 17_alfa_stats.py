"""
16_alfa_compare.py 결과를 바탕으로:
1. 이상치(outlier) 분석
2. 통과율 통계적 유의성 검증 (McNemar / Fisher / Binomial)
3. Test RMSE 페어드 t-test (시드별)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
n = len(X)
train_end = int(n * 0.8); val_end = int(n * 0.9)

X_train_raw = X[:train_end]; Y_train = Y[:train_end]
X_val_raw   = X[train_end:val_end]; Y_val = Y[train_end:val_end]
X_test_raw  = X[val_end:]; Y_test = Y[val_end:]

scaler_x = MinMaxScaler()
X_train = scaler_x.fit_transform(X_train_raw)
X_val   = scaler_x.transform(X_val_raw)
X_test  = scaler_x.transform(X_test_raw)

WINDOW = 10
def make_window(X, Y, w):
    Xw, Yw = [], []
    for i in range(w, len(X)):
        Xw.append(X[i-w:i]); Yw.append(Y[i])
    return np.array(Xw), np.array(Yw)

X_train_w, Y_train_w = make_window(X_train, Y_train, WINDOW)
X_val_w,   Y_val_w   = make_window(X_val,   Y_val,   WINDOW)
X_test_w,  Y_test_w  = make_window(X_test,  Y_test,  WINDOW)

zc_vr = np.sqrt(mean_squared_error(Y_val_w,  np.zeros(len(Y_val_w))))
zc_vm = mean_absolute_error(Y_val_w,  np.zeros(len(Y_val_w)))
zc_tr = np.sqrt(mean_squared_error(Y_test_w, np.zeros(len(Y_test_w))))
zc_tm = mean_absolute_error(Y_test_w, np.zeros(len(Y_test_w)))

# ── 모델 정의 ──────────────────────────────────────────────────────
input_shape  = (WINDOW, X_train.shape[1])
num_features = X_train.shape[1]

class FeatureAttention(Layer):
    def __init__(self, num_features, hidden_units, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.hidden_units = hidden_units
    def build(self, input_shape):
        self.feature_attention  = Dense(self.num_features, activation="softmax")
        self.hidden_to_features = Dense(self.num_features, activation="linear")
        super().build(input_shape)
    def call(self, inputs, **kwargs):
        attn   = self.feature_attention(inputs)
        proj   = self.hidden_to_features(inputs)
        cv     = K.sum(attn * proj, axis=-1)
        return cv, attn
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_features": self.num_features, "hidden_units": self.hidden_units})
        return cfg

def build_lstm(seed):
    tf.random.set_seed(seed); np.random.seed(seed)
    inp = Input(shape=input_shape)
    x   = LSTM(32, return_sequences=False)(inp)
    out = Dense(1)(x)
    m   = Model(inp, out); m.compile(optimizer="adam", loss="mse")
    return m

def build_temporal(seed):
    tf.random.set_seed(seed); np.random.seed(seed)
    inp  = Input(shape=input_shape)
    lo   = LSTM(32, return_sequences=True)(inp)
    sc   = Dense(1, activation='tanh')(lo)
    wt   = Softmax(axis=1)(sc)
    ctx  = Lambda(lambda x: K.sum(x * wt, axis=1))(lo)
    out  = Dense(1)(ctx)
    m    = Model(inp, out); m.compile(optimizer="adam", loss="mse")
    return m

def build_alfa(seed):
    tf.random.set_seed(seed); np.random.seed(seed)
    inp   = Input(shape=input_shape)
    lo    = LSTM(64, return_sequences=True)(inp)
    al    = FeatureAttention(num_features, 128)
    ctx,_ = al(lo)
    out   = Dense(1, activation="linear")(ctx)
    m     = Model(inp, out)
    adam  = tf.keras.optimizers.Adam(clipvalue=0.5, learning_rate=0.001,
                                     beta_1=0.9, beta_2=0.99, epsilon=1e-7, decay=0.001)
    m.compile(loss="mse", optimizer=adam)
    return m

# ── 20개 시드 실행 & 결과 저장 ─────────────────────────────────────
SEEDS = list(range(20))
builders = [("LSTM", build_lstm, 20),
            ("Temporal-Attn", build_temporal, 20),
            ("ALFA", build_alfa, 40)]

records = []
print("모델 학습 중...\n")
for seed in SEEDS:
    for name, builder, patience in builders:
        model = builder(seed)
        model.fit(X_train_w, Y_train_w,
                  validation_data=(X_val_w, Y_val_w),
                  epochs=300, batch_size=32,
                  callbacks=[EarlyStopping(monitor="val_loss", patience=patience,
                                           restore_best_weights=True)],
                  verbose=0)
        vp = model.predict(X_val_w,  verbose=0).flatten()
        tp = model.predict(X_test_w, verbose=0).flatten()
        vr = np.sqrt(mean_squared_error(Y_val_w,  vp))
        vm = mean_absolute_error(Y_val_w,  vp)
        tr = np.sqrt(mean_squared_error(Y_test_w, tp))
        tm = mean_absolute_error(Y_test_w, tp)
        passed = (vr < zc_vr and vm < zc_vm and tr < zc_tr and tm < zc_tm)
        records.append({"seed": seed, "model": name,
                        "val_rmse": vr, "val_mae": vm,
                        "test_rmse": tr, "test_mae": tm, "passed": passed})
        print(f"  seed={seed:>2} {name:<16} TestRMSE={tr:.4f}  {'✓' if passed else '✗'}")

df_res = pd.DataFrame(records)
df_res.to_csv("alfa_stats_results.csv", index=False)
print("\n결과 저장: alfa_stats_results.csv\n")

# ── 1. 이상치 분석 ─────────────────────────────────────────────────
print("="*65)
print("[ 이상치 분석 ]")
print(f"Zero Change Test RMSE: {zc_tr:.4f}원\n")

for name in ["LSTM", "Temporal-Attn", "ALFA"]:
    sub = df_res[df_res["model"] == name]["test_rmse"]
    q1, q3 = sub.quantile(0.25), sub.quantile(0.75)
    iqr = q3 - q1
    outliers = sub[sub > q3 + 1.5 * iqr]
    print(f"{name:<16} 평균:{sub.mean():.4f}  σ:{sub.std():.4f}  "
          f"이상치 {len(outliers)}개: {outliers.values.round(4).tolist()}")

print()

# ── 2. 통과율 통계 검증 ────────────────────────────────────────────
print("="*65)
print("[ 통과율 통계 검증 ]")

n_seeds = len(SEEDS)
pass_counts = {name: df_res[df_res["model"]==name]["passed"].sum()
               for name in ["LSTM", "Temporal-Attn", "ALFA"]}

for name, cnt in pass_counts.items():
    print(f"  {name:<16}: {cnt}/{n_seeds} ({cnt/n_seeds*100:.0f}%)")
print()

# Binomial test: ALFA vs 50% (랜덤보다 낫냐)
for name, cnt in pass_counts.items():
    res = stats.binomtest(cnt, n_seeds, p=0.5, alternative='greater')
    print(f"  Binomial test ({name} > 50%): p={res.pvalue:.4f}  "
          f"{'유의 *' if res.pvalue < 0.05 else '비유의'}")
print()

# Fisher's exact test: ALFA vs LSTM
lstm_pass = pass_counts["LSTM"]
alfa_pass = pass_counts["ALFA"]
table = [[alfa_pass, n_seeds - alfa_pass],
         [lstm_pass, n_seeds - lstm_pass]]
_, p_fisher = stats.fisher_exact(table, alternative='greater')
print(f"  Fisher's exact test (ALFA > LSTM): p={p_fisher:.4f}  "
      f"{'유의 *' if p_fisher < 0.05 else '비유의'}")
print()

# ── 3. 페어드 t-test: Test RMSE (같은 시드끼리) ────────────────────
print("="*65)
print("[ 페어드 t-test: Test RMSE (같은 시드 기준) ]")

for m1, m2 in [("ALFA", "LSTM"), ("ALFA", "Temporal-Attn"), ("LSTM", "Temporal-Attn")]:
    r1 = df_res[df_res["model"]==m1].sort_values("seed")["test_rmse"].values
    r2 = df_res[df_res["model"]==m2].sort_values("seed")["test_rmse"].values
    t, p = stats.ttest_rel(r1, r2)
    diff = np.mean(r1 - r2)
    print(f"  {m1} vs {m2:<16} mean_diff={diff:+.4f}원  p={p:.4f}  "
          f"{'유의 *' if p < 0.05 else '비유의'}")
print()

# ── 4. 결론 출력 ───────────────────────────────────────────────────
print("="*65)
print("[ 종합 결론 ]")
sig_alfa_50   = stats.binomtest(alfa_pass, n_seeds, p=0.5, alternative='greater').pvalue < 0.05
sig_alfa_lstm = p_fisher < 0.05
r_alfa = df_res[df_res["model"]=="ALFA"]["test_rmse"]
r_lstm = df_res[df_res["model"]=="LSTM"]["test_rmse"]
_, p_pair = stats.ttest_rel(r_alfa.sort_values().values, r_lstm.sort_values().values)

if sig_alfa_50 and sig_alfa_lstm:
    print("  → ALFA가 LSTM 대비 통계적으로 유의하게 우수 (p<0.05)")
elif sig_alfa_50:
    print("  → ALFA는 랜덤 대비 유의하게 통과하지만, LSTM 대비 우위는 비유의")
else:
    print("  → ALFA vs LSTM 차이가 통계적으로 유의하지 않음 (Meese-Rogoff 패턴 반복)")
