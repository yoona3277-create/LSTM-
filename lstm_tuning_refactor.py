import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt

from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
tf.random.set_seed(42)

TARGET = "비플럭스"
FEATURES = [
    "원수 탁도",
    "원수 pH",
    "rain",
    "temp",
    "원수 수온",
    "원수 TDS/전기전도도",
]
SEQ_LENS = [168]
HORIZONS = [12, 24]
TRAIN_RATIO = 0.8


def make_multi_sequences(df_x_scaled: pd.DataFrame, y_scaled: pd.Series, seq_len: int, horizon: int):
    """Build X:[N, seq_len, n_features], y:[N, horizon] for direct multi-step forecasting."""
    x_out, y_out, idx = [], [], []
    n = len(df_x_scaled)

    # 마지막 가능한 샘플까지 포함하려면 +1 필요
    for i in range(n - seq_len - horizon + 1):
        x_out.append(df_x_scaled.iloc[i : i + seq_len].values)
        y_out.append(y_scaled.iloc[i + seq_len : i + seq_len + horizon].values)
        idx.append(df_x_scaled.index[i + seq_len + horizon - 1])

    return (
        np.array(x_out, dtype="float32"),
        np.array(y_out, dtype="float32"),
        pd.DatetimeIndex(idx),
    )


def build_lstm_tuner(hp, input_shape, horizon):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(LSTM(units=hp.Choice("lstm_units_1", [64, 128]), return_sequences=True))
    model.add(Dropout(hp.Choice("dropout_1", [0.0, 0.1, 0.2])))

    model.add(LSTM(units=hp.Choice("lstm_units_2", [32, 64, 128]), return_sequences=False))
    model.add(Dropout(hp.Choice("dropout_2", [0.0, 0.1, 0.2])))

    model.add(Dense(units=hp.Choice("dense_units", [32, 64]), activation="relu"))
    model.add(Dense(horizon))

    lr = hp.Choice("learning_rate", [1e-3, 5e-4, 1e-4])
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return model


def tune_lstm_model(x_train, y_train, input_shape, horizon, project_name, max_trials=2):
    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_lstm_tuner(hp, input_shape, horizon),
        objective="val_loss",
        max_trials=max_trials,
        executions_per_trial=1,
        directory="keras_tuner_dir",
        project_name=project_name,
        overwrite=True,
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    tuner.search(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=20,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1,
        shuffle=False,
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hp)
    return best_model, best_hp


def build_datasets(data: pd.DataFrame, target=TARGET, features=None, seq_lens=None, horizons=None):
    features = features or FEATURES
    seq_lens = seq_lens or SEQ_LENS
    horizons = horizons or HORIZONS

    data = data.sort_index().copy()
    use_cols = list(dict.fromkeys(features + [target]))
    data[use_cols] = data[use_cols].interpolate(limit_direction="both")

    split_idx = int(len(data) * TRAIN_RATIO)
    df_train = data.iloc[:split_idx].copy()
    df_test = data.iloc[split_idx:].copy()

    scaler_x = RobustScaler()
    scaler_y = RobustScaler()

    x_train_scaled = scaler_x.fit_transform(df_train[features].values)
    x_test_scaled = scaler_x.transform(df_test[features].values)
    y_train_scaled = scaler_y.fit_transform(df_train[[target]].values)
    y_test_scaled = scaler_y.transform(df_test[[target]].values)

    df_train_scaled = pd.DataFrame(x_train_scaled, index=df_train.index, columns=features)
    df_test_scaled = pd.DataFrame(x_test_scaled, index=df_test.index, columns=features)
    y_train_series = pd.Series(y_train_scaled.flatten(), index=df_train.index)
    y_test_series = pd.Series(y_test_scaled.flatten(), index=df_test.index)

    datasets = {}
    for seq_len in seq_lens:
        for horizon in horizons:
            xtr, ytr, idx_tr = make_multi_sequences(df_train_scaled, y_train_series, seq_len, horizon)

            context_x = pd.concat([df_train_scaled.tail(seq_len), df_test_scaled], axis=0)
            context_y = pd.concat([y_train_series.tail(seq_len), y_test_series], axis=0)
            xte, yte, idx_te = make_multi_sequences(context_x, context_y, seq_len, horizon)

            keep = idx_te >= df_test.index[0]
            xte, yte, idx_te = xte[keep], yte[keep], idx_te[keep]

            datasets[(seq_len, horizon)] = {
                "X_train": xtr,
                "y_train": ytr,
                "idx_train": idx_tr,
                "X_test": xte,
                "y_test": yte,
                "idx_test": idx_te,
                "input_shape": (seq_len, len(features)),
            }
            print(f"[seq={seq_len}, horizon={horizon}] X_train={xtr.shape}, y_train={ytr.shape}")

    return datasets, scaler_x, scaler_y


def run_tuning_and_save(datasets, seq_lens=None, horizons=None, epochs=30, batch_size=32):
    seq_lens = seq_lens or SEQ_LENS
    horizons = horizons or HORIZONS

    best_hparams_summary = []
    for seq_len in seq_lens:
        for horizon in horizons:
            x_train = datasets[(seq_len, horizon)]["X_train"]
            y_train = datasets[(seq_len, horizon)]["y_train"]
            input_shape = datasets[(seq_len, horizon)]["input_shape"]

            print(f"\n🔍 Tuning LSTM for seq={seq_len}, horizon={horizon}")
            best_lstm, best_hp = tune_lstm_model(
                x_train,
                y_train,
                input_shape=input_shape,
                horizon=horizon,
                project_name=f"lstm_seq{seq_len}_h{horizon}",
            )

            print("✅ Best Hyperparameters")
            for key in [
                "lstm_units_1",
                "dropout_1",
                "lstm_units_2",
                "dropout_2",
                "dense_units",
                "learning_rate",
            ]:
                print(f"{key}:", best_hp.get(key))

            best_lstm.fit(
                x_train,
                y_train,
                validation_split=0.1,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[EarlyStopping(patience=7, restore_best_weights=True)],
                verbose=1,
                shuffle=False,
            )

            model_path = f"lstm_tuned_seq{seq_len}_h{horizon}.h5"
            best_lstm.save(model_path)
            print(f"✅ Saved: {model_path}")

            best_hparams_summary.append(
                {
                    "seq_len": seq_len,
                    "horizon": horizon,
                    "lstm_units_1": best_hp.get("lstm_units_1"),
                    "dropout_1": best_hp.get("dropout_1"),
                    "lstm_units_2": best_hp.get("lstm_units_2"),
                    "dropout_2": best_hp.get("dropout_2"),
                    "dense_units": best_hp.get("dense_units"),
                    "learning_rate": best_hp.get("learning_rate"),
                }
            )

    return pd.DataFrame(best_hparams_summary)


if __name__ == "__main__":
    raise SystemExit(
        "Use this module from a notebook/script: datasets, _, _ = build_datasets(data); "
        "best_hparams_df = run_tuning_and_save(datasets)"
    )
