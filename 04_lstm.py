import random                                                                                                                                      
import numpy as np                                                                                                                                 
import tensorflow as tf

random.seed(42)                                                                                                                                    
np.random.seed(42)
tf.random.set_seed(42)import numpy as np

import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
X_train = np.load("X_train.npy")
X_val   = np.load("X_val.npy")
X_test  = np.load("X_test.npy")
Y_train = np.load("Y_train.npy")
Y_val   = np.load("Y_val.npy")
Y_test  = np.load("Y_test.npy")
scaler_y = joblib.load("scaler_y.pkl")

# ① window 생성 함수 (과거 30일치를 묶어서 입력)
WINDOW = 30

def make_window(X, Y, window):
    Xw, Yw = [], []
    for i in range(window, len(X)):
        Xw.append(X[i-window:i])
        Yw.append(Y[i])
    return np.array(Xw), np.array(Yw)

X_train_w, Y_train_w = make_window(X_train, Y_train, WINDOW)
X_val_w,   Y_val_w   = make_window(X_val,   Y_val,   WINDOW)
X_test_w,  Y_test_w  = make_window(X_test,  Y_test,  WINDOW)

print(f"LSTM 입력 shape: {X_train_w.shape}")  # (샘플수, 30, 9)

# ② 모델 구성
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW, X_train.shape[1])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")
model.summary()

# ③ 학습
history = model.fit(
    X_train_w, Y_train_w,
    validation_data=(X_val_w, Y_val_w),
    epochs=50,
    batch_size=32,
    verbose=1
)

# ④ 예측 + 역정규화
pred = model.predict(X_test_w)
Y_real    = scaler_y.inverse_transform(Y_test_w.reshape(-1,1)).flatten()
pred_real = scaler_y.inverse_transform(pred).flatten()

# ⑤ 성능
rmse = np.sqrt(mean_squared_error(Y_real, pred_real))
mae  = mean_absolute_error(Y_real, pred_real)
r2   = r2_score(Y_real, pred_real)
print(f"\n===== LSTM 성능 =====")
print(f"RMSE: {rmse:.2f}원 / MAE: {mae:.2f}원 / R²: {r2:.4f}")

# ⑥ 시각화
plt.figure(figsize=(14,5))
plt.plot(Y_real,    label="실제 환율", color="steelblue")
plt.plot(pred_real, label="LSTM 예측", color="tomato", linestyle="--")
plt.title("LSTM — 실제 vs 예측 환율")
plt.legend()
plt.tight_layout()
plt.savefig("lstm_result.png")
plt.show()
