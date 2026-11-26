"""Cấu hình LSTNet theo quy ước chung của nhóm.

Các quy ước bắt buộc (xem README của nhóm):

- Target chính thức là ``nextPrice`` nhưng mô hình học
  ``delta = nextPrice - Price`` rồi cộng lại giá trước khi tính metric.
- Chia dữ liệu theo thời gian 80/20 (Train/Test), không shuffle.
- Trong Train: 90% đầu là ``train_fit`` (huấn luyện), 10% cuối là
  ``train_val`` (validation / EarlyStopping).
- Scaler và scaler của ``delta`` chỉ được fit trên ``train_fit``.
- Cửa sổ mặc định 30 ngày; dự đoán ``delta`` của ngày cuối cửa sổ.
- Metric tính trên giá USD: MAE, RMSE, MAPE, R², Directional Accuracy.

File dữ liệu: nhóm đặt tên ``Apple_Stock_ML_ready.csv``. Repo hiện có
``Apple_Stock_ready.csv`` với đúng 15 cột tương đương — chỉ cần đổi
``CONFIG["data"]`` nếu dùng file khác.
"""

# ---------------------------------------------------------------------------
# Dữ liệu & tiền xử lý
# ---------------------------------------------------------------------------
CONFIG = {
    "data": "Apple_Stock_ready.csv",
    "date_col": "Date",
    "price_col": "Price",
    "target_col": "NextPrice",        # file của nhóm có thể là "nextPrice"

    # Tập feature cơ sở (bỏ Date, NextPrice, Change_Pct vì trùng Return_1)
    "features": [
        "Price", "Open", "High", "Low", "Volume",
        "Return_1", "High_Low_Range", "Open_Close_Change",
    ],

    "train_ratio": 0.80,              # 80% Train / 20% Test
    "fit_ratio": 0.90,                # trong Train: 90% fit / 10% val
    "window": 30,                     # độ dài cửa sổ (ngày)
    "scaler": "standard",             # "standard" | "minmax"

    # --- Kiến trúc LSTNet ---
    # Các cấu hình thành phần tích chập
    "CNNFilters": 32,                 # 0 => bỏ nhánh CNN
    "CNNKernel": 6,

    # Cấu hình GRU và Skip-GRU
    "GRUUnits": 64,
    "SkipGRUUnits": 32,
    "skip": 5,                        # chu kỳ skip (1 tuần giao dịch); 0 => bỏ
    "dropout": 0.2,

    # Cấu hình nhánh AR (AutoRegressive) — học delta quá khứ để dự đoán delta hiện tại. Nếu highway=0 => bỏ nhánh AR.
    "highway": 7,                     # số delta quá khứ cho nhánh AR; 0 => bỏ
    "initializer": "glorot_uniform",

    # --- Huấn luyện ---
    "optimizer": "Adam",              # SGD | RMSprop | Adam
    "lr": 1e-3,
    "loss": "mse",
    "batchsize": 64,
    "epochs": 200,                    # EarlyStopping sẽ dừng sớm
    "earlystop_patience": 15,
    "earlystop_min_delta": 1e-5,

    # --- Tái lập kết quả ---
    "seeds": [42, 43, 44],            # chạy ít nhất 3 seed, báo cáo mean ± std

    # --- Đầu ra ---
    # None => không ghi file dự đoán. Chỉ đặt "predictions" khi thực sự cần
    # xuất CSV cho Weighted Average Ensemble.
    "pred_dir": None,
}

# ---------------------------------------------------------------------------
# Biểu đồ
# ---------------------------------------------------------------------------
PLOT = {
    "history": True,                  # vẽ loss/val_loss qua các epoch
    "prediction": True,               # vẽ nextPrice thực tế vs dự đoán
    "save_plot": None,                # None => chỉ hiển thị inline trong notebook
    "show": True,
    "dpi": 120,
    "figsize": (12, 4.5),
    "prediction_figsize": (14, 5),
}
