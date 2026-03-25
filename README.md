# LSTM-

`lstm_tuning_refactor.py`는 아래 순서로 바로 실행 가능한 형태로 정리되어 있습니다.

1. `datasets, scaler_X, scaler_y = build_datasets(data)`
2. `best_hparams_df = run_tuning_and_save(datasets)`

(여기서 `data`는 인덱스가 시간축인 `pandas.DataFrame`이어야 합니다.)
