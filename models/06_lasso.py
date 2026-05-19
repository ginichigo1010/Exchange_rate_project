import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
X_train = np.load("X_train.npy")
X_val   = np.load("X_val.npy")
X_test  = np.load("X_test.npy")
Y_train = np.load("Y_train.npy")
Y_val   = np.load("Y_val.npy")
Y_test  = np.load("Y_test.npy")
scaler_y = joblib.load("scaler_y.pkl")

feature_names = ["KRW", "DXY", "OIL", "KOSPI", "US10Y", "SP500", "VIX", "KOR_RATE", "US_CPI"]

# ── 1. Alpha 탐색 ──────────────────────────────────────────────
alphas = np.logspace(-4, 0, 50)
val_rmses = []

for alpha in alphas:
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train, Y_train)
    pred_val = model.predict(X_val)
    Y_val_real   = scaler_y.inverse_transform(Y_val.reshape(-1, 1)).flatten()
    pred_val_real = scaler_y.inverse_transform(pred_val.reshape(-1, 1)).flatten()
    val_rmses.append(np.sqrt(mean_squared_error(Y_val_real, pred_val_real)))

best_idx   = np.argmin(val_rmses)
best_alpha = alphas[best_idx]
print(f"최적 alpha: {best_alpha:.6f}  (val RMSE: {val_rmses[best_idx]:.2f}원)")

# ── 2. 최적 alpha로 재학습 & 테스트 평가 ───────────────────────
best_model = Lasso(alpha=best_alpha, max_iter=10000)
best_model.fit(X_train, Y_train)

pred_test = best_model.predict(X_test)
Y_real    = scaler_y.inverse_transform(Y_test.reshape(-1, 1)).flatten()
pred_real = scaler_y.inverse_transform(pred_test.reshape(-1, 1)).flatten()

rmse = np.sqrt(mean_squared_error(Y_real, pred_real))
mae  = mean_absolute_error(Y_real, pred_real)
r2   = r2_score(Y_real, pred_real)
print(f"[테스트] RMSE: {rmse:.2f}원 / MAE: {mae:.2f}원 / R²: {r2:.4f}")

# ── 3. 시각화 ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) Val RMSE vs Alpha
axes[0].plot(alphas, val_rmses, color="steelblue")
axes[0].axvline(best_alpha, color="tomato", linestyle="--", label=f"최적 α={best_alpha:.5f}")
axes[0].set_xscale("log")
axes[0].set_xlabel("Alpha (log scale)")
axes[0].set_ylabel("Val RMSE (원)")
axes[0].set_title("Alpha 탐색 곡선")
axes[0].legend()

# (b) 실제 vs 예측 (테스트)
axes[1].plot(Y_real,    label="실제 환율", color="steelblue")
axes[1].plot(pred_real, label="예측 환율", color="tomato", linestyle="--")
axes[1].set_title(f"LASSO — 실제 vs 예측\nRMSE={rmse:.2f}원  R²={r2:.4f}")
axes[1].set_xlabel("테스트 시점")
axes[1].set_ylabel("환율 (원)")
axes[1].legend()

# (c) 피처 계수
coefs = best_model.coef_
colors = ["tomato" if c != 0 else "lightgray" for c in coefs]
bars = axes[2].barh(feature_names, coefs, color=colors)
axes[2].axvline(0, color="black", linewidth=0.8)
axes[2].set_title("피처 계수 (회색 = 제거된 변수)")
axes[2].set_xlabel("계수 값")
for bar, c in zip(bars, coefs):
    label = f"{c:.4f}" if c != 0 else "0 (제거)"
    axes[2].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 label, va="center", fontsize=8)

plt.tight_layout()
plt.savefig("lasso_result.png")
plt.show()

# ── 4. 계수 요약 출력 ──────────────────────────────────────────
print("\n[피처 계수]")
for name, coef in zip(feature_names, coefs):
    status = f"{coef:.6f}" if coef != 0 else "0.000000  ← 제거됨"
    print(f"  {name:12s}: {status}")
zero_count = np.sum(coefs == 0)
print(f"\n총 {len(feature_names)}개 중 {zero_count}개 제거, {len(feature_names)-zero_count}개 생존")
