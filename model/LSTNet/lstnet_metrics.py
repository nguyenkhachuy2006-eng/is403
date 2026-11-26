"""Metric thống nhất của nhóm, tính trên giá USD."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred, price_today):
    """Tính MAE, RMSE, MAPE (%), R² và Directional Accuracy (%).

    Tham số
    -------
    y_true : ``nextPrice`` thực tế (USD).
    y_pred : ``nextPrice`` dự đoán (USD), đã cộng lại từ ``delta``.
    price_today : ``Price`` của ngày t, dùng để xác định hướng tăng/giảm.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    price_today = np.asarray(price_today, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    directional = np.mean(
        np.sign(y_pred - price_today) == np.sign(y_true - price_today)
    ) * 100

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2),
        "DirAcc": float(directional),
    }


def metrics_table(results):
    """Tổng hợp mean ± std của các metric qua nhiều seed.

    ``results`` là DataFrame với các cột dạng ``val_MAE``, ``test_RMSE``, ...
    """
    import pandas as pd

    metric_cols = [
        column for column in results.columns
        if column.startswith(("val_", "test_"))
    ]
    summary = results[metric_cols].agg(["mean", "std"]).T
    summary["mean ± std"] = summary.apply(
        lambda row: "%.4f ± %.4f" % (row["mean"], row["std"])
        if not np.isnan(row["std"]) else "%.4f" % row["mean"],
        axis=1,
    )
    return summary
