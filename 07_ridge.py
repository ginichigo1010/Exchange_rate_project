import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
X_train  = np.load("X_train_v2.npy")
X_val    = np.load("X_val_v2.npy")
X_test   = np.load("X_test_v2.npy")
Y_train  = np.load("Y_train_v2.npy")
Y_val    = np.load("Y_val_v2.npy")
Y_test   = np.load("Y_test_v2.npy")
scaler_y = joblib.load("scaler_y_v2.pkl")
features = joblib.load("features_v2.pkl")

# ── 1. Alpha 탐색 ──────────────────────────────────────────────
alphas = np.logspace(-3, 3, 100)
val_rmses = []

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train, Y_train)
    pred_val      = model.predict(X_val)
    Y_val_real    = scaler_y.inverse_transform(Y_val.reshape(-1, 1)).flatten()
    pred_val_real = scaler_y.inverse_transform(pred_val.reshape(-1, 1)).flatten()
    val_rmses.append(np.sqrt(mean_squared_error(Y_val_real, pred_val_real)))

best_idx   = np.argmin(val_rmses)
best_alpha = alphas[best_idx]
print(f"최적 alpha: {best_alpha:.4f}  (val RMSE: {val_rmses[best_idx]:.2f}원)")

# ── 2. 최적 alpha로 재학습 & 테스트 평가 ───────────────────────
best_model = Ridge(alpha=best_alpha)
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
axes[0].axvline(best_alpha, color="tomato", linestyle="--", label=f"최적 α={best_alpha:.4f}")
axes[0].set_xscale("log")
axes[0].set_xlabel("Alpha (log scale)")
axes[0].set_ylabel("Val RMSE (원)")
axes[0].set_title("Alpha 탐색 곡선")
axes[0].legend()

# (b) 실제 vs 예측 (테스트)
axes[1].plot(Y_real,    label="실제 환율", color="steelblue")
axes[1].plot(pred_real, label="예측 환율", color="mediumpurple", linestyle="--")
axes[1].set_title(f"Ridge — 실제 vs 예측\nRMSE={rmse:.2f}원  R²={r2:.4f}")
axes[1].set_xlabel("테스트 시점")
axes[1].set_ylabel("환율 (원)")
axes[1].legend()

# (c) 피처 계수 (Ridge는 0이 없으므로 크기 기준 정렬)
coefs   = best_model.coef_
sort_idx = np.argsort(np.abs(coefs))
sorted_features = [features[i] for i in sort_idx]
sorted_coefs    = coefs[sort_idx]
colors = ["tomato" if c > 0 else "steelblue" for c in sorted_coefs]

axes[2].barh(sorted_features, sorted_coefs, color=colors)
axes[2].axvline(0, color="black", linewidth=0.8)
axes[2].set_title("피처 계수 (절댓값 오름차순, 빨강=양/파랑=음)")
axes[2].set_xlabel("계수 값")

plt.tight_layout()
plt.savefig("ridge_result.png")
plt.show()

# ── 4. 계수 요약 출력 ──────────────────────────────────────────
print("\n[피처 계수] (절댓값 내림차순)")
for i in np.argsort(np.abs(coefs))[::-1]:
    print(f"  {features[i]:18s}: {coefs[i]:+.6f}")
