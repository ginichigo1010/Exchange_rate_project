import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 로드 & delta 타깃 생성 ──────────────────────────────────
df = pd.read_csv("raw_data_v2.csv", index_col="Date", parse_dates=True)
df = df.ffill().dropna()

# 파생변수 (02_preprocess_v2.py 동일)
df["STOCK_DIFF"]     = df["KOSPI"] - df["SP500"]
df["MA_DIFF"]        = df["MA5"]   - df["MA20"]
df["JPY_KRW_SPREAD"] = df["JPY"]   - df["KRW"] / 10
df = df.dropna()

# 타깃: 내일 환율 - 오늘 환율 (1일 변화량)
df["TARGET"] = df["KRW"].shift(-1) - df["KRW"]
df = df.dropna()

features = [
    "KRW", "DXY", "OIL", "KOSPI", "US10Y", "VIX", "SP500", "STOCK_DIFF",
    "JPY", "JPY_KRW_SPREAD", "MA5", "MA20", "RSI", "MA_DIFF",
    "SOX", "KR_RATE", "US_CPI",
]

X = df[features].values
Y = df["TARGET"].values          # 단위: 원 (역정규화 없이 그대로 사용)
KRW_today = df["KRW"].values     # Zero Change 평가용

# ── 분할 (8:1:1, 시간 순서 유지) ──────────────────────────────────
n         = len(X)
train_end = int(n * 0.8)
val_end   = int(n * 0.9)

X_train_raw = X[:train_end];  Y_train = Y[:train_end]
X_val_raw   = X[train_end:val_end]; Y_val   = Y[train_end:val_end]
X_test_raw  = X[val_end:];   Y_test  = Y[val_end:]
KRW_val     = KRW_today[train_end:val_end]
KRW_test    = KRW_today[val_end:]

print(f"훈련: {len(X_train_raw)}일 / 검증: {len(X_val_raw)}일 / 테스트: {len(X_test_raw)}일")

# ── X만 정규화 (Y는 원 단위 그대로) ───────────────────────────────
scaler_x = MinMaxScaler()
X_train  = scaler_x.fit_transform(X_train_raw)
X_val    = scaler_x.transform(X_val_raw)
X_test   = scaler_x.transform(X_test_raw)

# ── baseline 모델 ──────────────────────────────────────────────────

# 1. Zero Change: 변화 없음 (delta = 0)
zc_val  = np.zeros(len(Y_val))
zc_test = np.zeros(len(Y_test))

# 2. Train Mean: 훈련 구간 평균 변화량
train_mean_delta = Y_train.mean()
tm_val  = np.full(len(Y_val),  train_mean_delta)
tm_test = np.full(len(Y_test), train_mean_delta)
print(f"훈련 구간 평균 변화량: {train_mean_delta:+.4f}원")

# 3. KRW Lag Linear: 과거 KRW값만 쓰는 선형회귀
krw_idx  = features.index("KRW")
krw_lag  = LinearRegression()
krw_lag.fit(X_train[:, [krw_idx]], Y_train)
kl_val   = krw_lag.predict(X_val[:, [krw_idx]])
kl_test  = krw_lag.predict(X_test[:, [krw_idx]])

# ── 비교 모델 ──────────────────────────────────────────────────────

# 선형회귀
lr = LinearRegression()
lr.fit(X_train, Y_train)
lr_val  = lr.predict(X_val)
lr_test = lr.predict(X_test)

# LASSO
alphas = np.logspace(-4, 0, 50)
lasso_val_rmses = []
for a in alphas:
    m = Lasso(alpha=a, max_iter=10000)
    m.fit(X_train, Y_train)
    lasso_val_rmses.append(np.sqrt(mean_squared_error(Y_val, m.predict(X_val))))
best_lasso_alpha = alphas[np.argmin(lasso_val_rmses)]
print(f"LASSO 최적 alpha: {best_lasso_alpha:.6f}")

lasso = Lasso(alpha=best_lasso_alpha, max_iter=10000)
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

# ── 평가 ──────────────────────────────────────────────────────────
def metrics(pred, actual):
    return (
        np.sqrt(mean_squared_error(actual, pred)),
        mean_absolute_error(actual, pred),
        r2_score(actual, pred),
    )

models = [
    ("Zero Change",    zc_val,    zc_test),
    ("Train Mean",     tm_val,    tm_test),
    ("KRW Lag Linear", kl_val,    kl_test),
    ("선형회귀",        lr_val,    lr_test),
    ("LASSO",          lasso_val, lasso_test),
    ("Ridge",          ridge_val, ridge_test),
]

zc_vr, zc_vm, _ = metrics(zc_val,  Y_val)
zc_tr, zc_tm, _ = metrics(zc_test, Y_test)

print("\n" + "="*75)
print(f"{'모델':<16} {'Val RMSE':>9} {'Val MAE':>9} {'Test RMSE':>10} {'Test MAE':>9}  gate")
print("-"*75)
rows = []
for name, val_pred, test_pred in models:
    vr, vm, _ = metrics(val_pred,  Y_val)
    tr, tm, _ = metrics(test_pred, Y_test)
    passed = (vr < zc_vr and vm < zc_vm and tr < zc_tr and tm < zc_tm)
    gate = "✓ 통과" if passed else "✗ 미통과"
    print(f"{name:<16} {vr:>8.2f}원 {vm:>8.2f}원 {tr:>9.2f}원 {tm:>8.2f}원  {gate}")
    rows.append((name, vr, vm, tr, tm, passed))
print("="*75)
print("gate 기준: Val·Test RMSE·MAE 모두 Zero Change(delta=0)보다 낮아야 통과\n")

# ── 시각화 ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) Test RMSE 막대 비교
names     = [r[0] for r in rows]
test_rmse = [r[3] for r in rows]
colors = []
for r in rows:
    if r[0] in ("Zero Change", "Train Mean", "KRW Lag Linear"):
        colors.append("lightcoral")
    elif r[5]:
        colors.append("mediumseagreen")
    else:
        colors.append("steelblue")

bars = axes[0].bar(names, test_rmse, color=colors)
axes[0].axhline(zc_tr, color="red", linestyle="--", linewidth=1.2,
                label=f"Zero Change ({zc_tr:.2f}원)")
for bar, val in zip(bars, test_rmse):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8.5)
axes[0].set_title("모델별 Test RMSE — delta 예측\n(빨강=baseline, 초록=gate 통과, 파랑=미통과)")
axes[0].set_ylabel("RMSE (원)")
axes[0].tick_params(axis='x', rotation=20)
axes[0].legend()

# (b) 테스트 구간 실제 delta vs 예측
axes[1].plot(Y_test,    label="실제 변화량",   color="steelblue",      linewidth=1.2)
axes[1].plot(kl_test,  label="KRW Lag",      color="orange",         linestyle="--", linewidth=1.0)
axes[1].plot(lasso_test, label="LASSO",      color="mediumseagreen", linestyle="--", linewidth=1.0)
axes[1].plot(lr_test,  label="선형회귀",      color="tomato",         linestyle=":",  linewidth=1.0)
axes[1].axhline(0, color="lightcoral", linestyle="-", linewidth=1.0, label="Zero Change (0)")
axes[1].set_title("테스트 구간 실제 변화량 vs 예측")
axes[1].set_xlabel("테스트 시점")
axes[1].set_ylabel("변화량 (원)")
axes[1].legend()

plt.tight_layout()
plt.savefig("delta_compare.png", dpi=150)
plt.show()
print("저장 완료: delta_compare.png")
