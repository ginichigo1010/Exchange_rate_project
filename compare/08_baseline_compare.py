import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 로드 ────────────────────────────────────────────────────
scaler_x = joblib.load("scaler_x_v2.pkl")
scaler_y = joblib.load("scaler_y_v2.pkl")
features  = joblib.load("features_v2.pkl")

X_train = np.load("X_train_v2.npy")
X_val   = np.load("X_val_v2.npy")
X_test  = np.load("X_test_v2.npy")
Y_train = np.load("Y_train_v2.npy")
Y_val   = np.load("Y_val_v2.npy")
Y_test  = np.load("Y_test_v2.npy")

Y_real       = scaler_y.inverse_transform(Y_test.reshape(-1, 1)).flatten()
Y_val_real   = scaler_y.inverse_transform(Y_val.reshape(-1, 1)).flatten()
Y_train_real = scaler_y.inverse_transform(Y_train.reshape(-1, 1)).flatten()

X_test_real  = scaler_x.inverse_transform(X_test)
X_val_real   = scaler_x.inverse_transform(X_val)
krw_idx      = features.index("KRW")

# ── baseline 모델 ──────────────────────────────────────────────────

# 1. Zero Change: 오늘 환율 = 내일 환율 (가장 단순한 기준선)
zc_val  = X_val_real[:, krw_idx]
zc_test = X_test_real[:, krw_idx]

# 2. Train Mean: 훈련 구간 평균값으로 예측
tm_val  = np.full(len(Y_val_real),  Y_train_real.mean())
tm_test = np.full(len(Y_real),      Y_train_real.mean())

# 3. KRW Lag Linear: 과거 KRW 값만 사용하는 선형회귀
krw_lag = LinearRegression()
krw_lag.fit(X_train[:, [krw_idx]], Y_train)
kl_val  = scaler_y.inverse_transform(krw_lag.predict(X_val[:, [krw_idx]]).reshape(-1, 1)).flatten()
kl_test = scaler_y.inverse_transform(krw_lag.predict(X_test[:, [krw_idx]]).reshape(-1, 1)).flatten()

# ── 비교 모델 (v2 데이터, val 기준 튜닝) ──────────────────────────

# 선형회귀
lr = LinearRegression()
lr.fit(X_train, Y_train)
lr_val  = scaler_y.inverse_transform(lr.predict(X_val).reshape(-1, 1)).flatten()
lr_test = scaler_y.inverse_transform(lr.predict(X_test).reshape(-1, 1)).flatten()

# LASSO — val RMSE 기준 alpha 탐색
alphas = np.logspace(-4, 0, 50)
lasso_val_rmses = []
for a in alphas:
    m = Lasso(alpha=a, max_iter=10000)
    m.fit(X_train, Y_train)
    pv = scaler_y.inverse_transform(m.predict(X_val).reshape(-1, 1)).flatten()
    lasso_val_rmses.append(np.sqrt(mean_squared_error(Y_val_real, pv)))
best_lasso_alpha = alphas[np.argmin(lasso_val_rmses)]
print(f"LASSO 최적 alpha: {best_lasso_alpha:.6f}")

lasso = Lasso(alpha=best_lasso_alpha, max_iter=10000)
lasso.fit(X_train, Y_train)
lasso_val  = scaler_y.inverse_transform(lasso.predict(X_val).reshape(-1, 1)).flatten()
lasso_test = scaler_y.inverse_transform(lasso.predict(X_test).reshape(-1, 1)).flatten()

# Ridge — val RMSE 기준 alpha 탐색
ridge_alphas = np.logspace(-3, 3, 100)
ridge_val_rmses = []
for a in ridge_alphas:
    m = Ridge(alpha=a)
    m.fit(X_train, Y_train)
    pv = scaler_y.inverse_transform(m.predict(X_val).reshape(-1, 1)).flatten()
    ridge_val_rmses.append(np.sqrt(mean_squared_error(Y_val_real, pv)))
best_ridge_alpha = ridge_alphas[np.argmin(ridge_val_rmses)]
print(f"Ridge  최적 alpha: {best_ridge_alpha:.4f}")

ridge = Ridge(alpha=best_ridge_alpha)
ridge.fit(X_train, Y_train)
ridge_val  = scaler_y.inverse_transform(ridge.predict(X_val).reshape(-1, 1)).flatten()
ridge_test = scaler_y.inverse_transform(ridge.predict(X_test).reshape(-1, 1)).flatten()

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

# baseline gate 기준: Zero Change val/test RMSE·MAE 모두 이겨야 통과
zc_val_rmse,  zc_val_mae,  _ = metrics(zc_val,  Y_val_real)
zc_test_rmse, zc_test_mae, _ = metrics(zc_test, Y_real)

print("\n" + "="*75)
print(f"{'모델':<16} {'Val RMSE':>9} {'Val MAE':>9} {'Test RMSE':>10} {'Test MAE':>9}  gate")
print("-"*75)
rows = []
for name, val_pred, test_pred in models:
    vr, vm, _ = metrics(val_pred,  Y_val_real)
    tr, tm, _ = metrics(test_pred, Y_real)
    passed = (vr < zc_val_rmse and vm < zc_val_mae and
              tr < zc_test_rmse and tm < zc_test_mae)
    gate = "✓ 통과" if passed else "✗ 미통과"
    print(f"{name:<16} {vr:>8.2f}원 {vm:>8.2f}원 {tr:>9.2f}원 {tm:>8.2f}원  {gate}")
    rows.append((name, vr, vm, tr, tm, passed))
print("="*75)
print("gate 기준: Val·Test RMSE·MAE 모두 Zero Change보다 낮아야 통과\n")

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
axes[0].axhline(zc_test_rmse, color="red", linestyle="--", linewidth=1.2,
                label=f"Zero Change ({zc_test_rmse:.2f}원)")
for bar, val in zip(bars, test_rmse):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8.5)
axes[0].set_title("모델별 Test RMSE\n(빨강=baseline, 초록=gate 통과, 파랑=미통과)")
axes[0].set_ylabel("RMSE (원)")
axes[0].tick_params(axis='x', rotation=20)
axes[0].legend()

# (b) 테스트 구간 실제 vs 예측
axes[1].plot(Y_real,    label="실제 환율",    color="steelblue",      linewidth=1.8)
axes[1].plot(zc_test,  label="Zero Change", color="lightcoral",     linestyle="--", linewidth=1.2)
axes[1].plot(kl_test,  label="KRW Lag",     color="orange",         linestyle="--", linewidth=1.2)
axes[1].plot(lasso_test, label="LASSO",     color="mediumseagreen", linestyle="--", linewidth=1.2)
axes[1].plot(lr_test,  label="선형회귀",     color="tomato",         linestyle=":",  linewidth=1.2)
axes[1].set_title("테스트 구간 실제 vs 예측")
axes[1].set_xlabel("테스트 시점")
axes[1].set_ylabel("환율 (원)")
axes[1].legend()

plt.tight_layout()
plt.savefig("baseline_compare.png", dpi=150)
plt.show()
print("저장 완료: baseline_compare.png")
