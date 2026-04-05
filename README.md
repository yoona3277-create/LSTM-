# LSTM-RO fouling prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import math

from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================================================
# seed
# =========================================================
np.random.seed(42)
tf.random.set_seed(42)

# =========================================================
# 설정
# =========================================================
TARGET = '비플럭스'
FEATURES = ['원수 탁도', '원수 pH', '원수 수온', '원수 TDS/전기전도도', 'cycle_id', 'cycle_index']

SEQ_LENS = [168]      # 입력: 지난 7일
HORIZONS = [12, 24]   # 출력: 12개 / 24개 예측
TRAIN_RATIO = 0.8
EPS = 1e-5

# =========================================================
# 데이터 정리
# =========================================================
data = data.sort_index().copy()

use_cols = FEATURES + [TARGET]
use_cols = list(dict.fromkeys(use_cols))

data[use_cols] = data[use_cols].interpolate(limit_direction='both')

n_total = len(data)
split_idx = int(n_total * TRAIN_RATIO)

df_tr = data.iloc[:split_idx].copy()
df_te = data.iloc[split_idx:].copy()

# RobustScaler 제거
# 원본 값 그대로 사용
df_tr_s = df_tr[FEATURES].copy()
df_te_s = df_te[FEATURES].copy()

y_tr_s = df_tr[TARGET].copy()
y_te_s = df_te[TARGET].copy()

print("train_df shape:", df_tr.shape)
print("test_df shape:", df_te.shape)

# =========================================================
# RevIN-style 정규화 / 역정규화 함수
# =========================================================
def revin_norm_3d(x, eps=1e-5):
    """
    x: (seq_len, n_features)
    각 feature별로 seq_len 축 기준 mean/std 계산
    """
    mean = x.mean(axis=0, keepdims=True)                 # (1, n_features)
    std = x.std(axis=0, keepdims=True) + eps            # (1, n_features)
    x_norm = (x - mean) / std
    return x_norm, mean, std


def revin_norm_1d_from_history(y_hist, y_future, eps=1e-5):
    """
    y_hist: (seq_len,)
    y_future: (horizon,)

    과거 y_hist의 mean/std를 사용해서 future y를 정규화
    """
    y_mean = y_hist.mean()
    y_std = y_hist.std() + eps
    y_future_norm = (y_future - y_mean) / y_std
    return y_future_norm, y_mean, y_std


def revin_denorm_y(y_norm, y_mean, y_std):
    """
    y_norm: (n_samples, horizon) 또는 (horizon,)
    y_mean: (n_samples, 1) 또는 scalar
    y_std : (n_samples, 1) 또는 scalar
    """
    return y_norm * y_std + y_mean

# =========================================================
# Multi-step sequence 생성 (RevIN 버전)
# =========================================================
def make_multi_sequences(df_X, y_raw, seq_len, horizon, eps=1e-5):
    X, y, idx = [], [], []
    y_mean_list, y_std_list = [], []

    n = len(df_X)

    for i in range(n - seq_len - horizon + 1):
        # 입력 X 시퀀스
        x_seq = df_X.iloc[i:i+seq_len].values.astype('float32')  # (seq_len, n_features)

        # target history / future
        y_hist = y_raw.iloc[i:i+seq_len].values.astype('float32')                    # (seq_len,)
        y_future = y_raw.iloc[i+seq_len:i+seq_len+horizon].values.astype('float32') # (horizon,)

        # X: feature-wise RevIN
        x_seq_norm, _, _ = revin_norm_3d(x_seq, eps=eps)

        # y: history target 기준 RevIN
        y_future_norm, y_mean, y_std = revin_norm_1d_from_history(y_hist, y_future, eps=eps)

        X.append(x_seq_norm)
        y.append(y_future_norm)
        idx.append(df_X.index[i + seq_len + horizon - 1])

        y_mean_list.append([y_mean])   # (n_samples, 1) 맞추기
        y_std_list.append([y_std])

    return (
        np.array(X, dtype='float32'),
        np.array(y, dtype='float32'),
        pd.DatetimeIndex(idx),
        np.array(y_mean_list, dtype='float32'),
        np.array(y_std_list, dtype='float32')
    )

# =========================================================
# Dataset 구성
# =========================================================
datasets = {}

for L in SEQ_LENS:
    for H in HORIZONS:
        Xtr_seq, ytr_seq, idx_tr_seq, ytr_mean, ytr_std = make_multi_sequences(
            df_tr_s, y_tr_s, L, H, eps=EPS
        )
        Xte_seq, yte_seq, idx_te_seq, yte_mean, yte_std = make_multi_sequences(
            df_te_s, y_te_s, L, H, eps=EPS
        )

        datasets[(L, H)] = {
            "X_train": Xtr_seq,
            "y_train": ytr_seq,
            "idx_train": idx_tr_seq,
            "X_test": Xte_seq,
            "y_test": yte_seq,
            "idx_test": idx_te_seq,
            "y_train_mean": ytr_mean,
            "y_train_std": ytr_std,
            "y_test_mean": yte_mean,
            "y_test_std": yte_std,
            "input_shape": (L, len(FEATURES)),
            "seq_len": L,
            "horizon": H
        }

        print(f"[seq={L}, horizon={H}] -> X_train={Xtr_seq.shape}, y_train={ytr_seq.shape}")
        print(f"[seq={L}, horizon={H}] -> X_test ={Xte_seq.shape}, y_test ={yte_seq.shape}")

# =========================================================
# 모델 구조
# =========================================================
def build_lstm(input_shape, horizon):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(32, kernel_regularizer=l2(0.01), return_sequences=False),
        Dropout(0.4),
        Dense(16, activation='relu'),
        Dense(horizon)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    return model

# =========================================================
# 학습 + 저장
# =========================================================
for L in SEQ_LENS:
    for H in HORIZONS:
        Xtr = datasets[(L, H)]["X_train"]
        ytr = datasets[(L, H)]["y_train"]
        input_shape = datasets[(L, H)]["input_shape"]

        print(f"\n=== Training LSTM (seq={L}, horizon={H}) ===")

        lstm = build_lstm(input_shape, H)
        lstm.fit(
            Xtr, ytr,
            validation_split=0.1,
            epochs=30,
            batch_size=32,
            callbacks=[EarlyStopping(patience=7, restore_best_weights=True)],
            verbose=1
        )

        save_name = f"lstm_seq{L}_h{H}.h5"
        lstm.save(save_name)
        print(f"✅ Saved: {save_name}")

# =========================================================
# RevIN 역변환
# =========================================================
def inv_scale_y(arr_s, y_mean, y_std):
    return revin_denorm_y(arr_s, y_mean, y_std)

# =========================================================
# 성능 계산 함수 (multi-step)
# =========================================================
def metrics_multistep(y_true, y_pred):
    H = y_true.shape[1]
    rmse_list, mae_list, r2_list = [], [], []

    for t in range(H):
        rmse_list.append(math.sqrt(mean_squared_error(y_true[:, t], y_pred[:, t])))
        mae_list.append(mean_absolute_error(y_true[:, t], y_pred[:, t]))
        r2_list.append(r2_score(y_true[:, t], y_pred[:, t]))

    return {
        "R2_mean": np.mean(r2_list),
        "RMSE_mean": np.mean(rmse_list),
        "MAE_mean": np.mean(mae_list),
        "R2_each": r2_list,
        "RMSE_each": rmse_list,
        "MAE_each": mae_list
    }

# =========================================================
# Scatter plot
# =========================================================
def plot_scatter_multistep(y_true, y_pred, seq_len, horizon, model_type, dataset_type):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true.flatten(), y_pred.flatten(), alpha=0.3, edgecolors='k')
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], 'r--')
    plt.title(f"{model_type} Scatter ({dataset_type})  seq={seq_len}, horizon={horizon}")
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.grid()
    plt.show()

# =========================================================
# Time-series plot
# =========================================================
def plot_series_multistep(y_true, y_pred, idx, seq_len, horizon, model_type, dataset_type):
    plt.figure(figsize=(14, 5))
    plt.plot(idx, y_true[:, 0], label="True(t+1)", color="black")
    plt.plot(idx, y_pred[:, 0], label="Pred(t+1)", color="blue", alpha=0.6)
    plt.title(f"{model_type} Time Series ({dataset_type}) seq={seq_len}, horizon={horizon}, step=t+1")
    plt.grid()
    plt.legend()
    plt.show()

# =========================================================
# 결과 저장 리스트
# =========================================================
performance_table_rows = []

# =========================================================
# 반복 평가
# =========================================================
for L in SEQ_LENS:
    for H in HORIZONS:
        print(f"\n\n=== 🔍 Evaluating Models (seq={L}, horizon={H}) ===")

        Xtr = datasets[(L, H)]["X_train"]
        ytr = datasets[(L, H)]["y_train"]
        Xte = datasets[(L, H)]["X_test"]
        yte = datasets[(L, H)]["y_test"]
        idx_tr = datasets[(L, H)]["idx_train"]
        idx_te = datasets[(L, H)]["idx_test"]

        ytr_mean = datasets[(L, H)]["y_train_mean"]
        ytr_std = datasets[(L, H)]["y_train_std"]
        yte_mean = datasets[(L, H)]["y_test_mean"]
        yte_std = datasets[(L, H)]["y_test_std"]

        # 정답 역정규화
        y_tr_true = inv_scale_y(ytr, ytr_mean, ytr_std)
        y_te_true = inv_scale_y(yte, yte_mean, yte_std)

        # 모델 로드
        lstm_model = load_model(f"lstm_seq{L}_h{H}.h5", compile=False)

        # TRAIN 예측 -> 역정규화
        lstm_tr_pred_norm = lstm_model.predict(Xtr, verbose=0)
        lstm_tr_pred = inv_scale_y(lstm_tr_pred_norm, ytr_mean, ytr_std)
        lstm_tr_m = metrics_multistep(y_tr_true, lstm_tr_pred)

        # TEST 예측 -> 역정규화
        lstm_te_pred_norm = lstm_model.predict(Xte, verbose=0)
        lstm_te_pred = inv_scale_y(lstm_te_pred_norm, yte_mean, yte_std)
        lstm_te_m = metrics_multistep(y_te_true, lstm_te_pred)

        # 그림
        plot_scatter_multistep(y_tr_true, lstm_tr_pred, L, H, "LSTM", "Train")
        plot_series_multistep(y_tr_true, lstm_tr_pred, idx_tr, L, H, "LSTM", "Train")

        plot_scatter_multistep(y_te_true, lstm_te_pred, L, H, "LSTM", "Test")
        plot_series_multistep(y_te_true, lstm_te_pred, idx_te, L, H, "LSTM", "Test")

        # 표 기록
        performance_table_rows.append({
            "Horizon": H,
            "LSTM_Train_R2": lstm_tr_m["R2_mean"],
            "LSTM_Test_R2": lstm_te_m["R2_mean"],
            "LSTM_Train_RMSE": lstm_tr_m["RMSE_mean"],
            "LSTM_Test_RMSE": lstm_te_m["RMSE_mean"],
            "LSTM_Train_MAE": lstm_tr_m["MAE_mean"],
            "LSTM_Test_MAE": lstm_te_m["MAE_mean"]
        })

# =========================================================
# 최종 표 출력 및 저장
# =========================================================
df_table = pd.DataFrame(performance_table_rows).sort_values("Horizon")
df_table = df_table.round(3)

print("\n\n===== 📊 Performance Summary Table (TRAIN + TEST) =====")
print(df_table)

df_table.to_csv("model_performance_table_train_test_revin.csv", index=False)
print("\n✅ Saved: model_performance_table_train_test_revin.csv")
