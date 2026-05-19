import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

X_train = np.load("X_train.npy")
X_test  = np.load("X_test.npy")
Y_train = np.load("Y_train.npy")
Y_test  = np.load("Y_test.npy")
scaler_y = joblib.load("scaler_y.pkl")

# 학습
model = LinearRegression()
model.fit(X_train, Y_train)

# 예측 + 역정규화
pred = model.predict(X_test)
Y_real    = scaler_y.inverse_transform(Y_test.reshape(-1,1)).flatten()
pred_real = scaler_y.inverse_transform(pred.reshape(-1,1)).flatten()

# 성능
rmse = np.sqrt(mean_squared_error(Y_real, pred_real))
mae  = mean_absolute_error(Y_real, pred_real)
r2   = r2_score(Y_real, pred_real)
print(f"RMSE: {rmse:.2f}원 / MAE: {mae:.2f}원 / R²: {r2:.4f}")

# 시각화
plt.figure(figsize=(14,5))
plt.plot(Y_real,    label="실제 환율", color="steelblue")
plt.plot(pred_real, label="예측 환율", color="tomato", linestyle="--")
plt.title("선형회귀 — 실제 vs 예측")
plt.legend()
plt.tight_layout()
plt.savefig("linear_result.png")
plt.show()
