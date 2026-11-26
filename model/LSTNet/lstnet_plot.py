"""Vẽ lịch sử huấn luyện và so sánh nextPrice thực tế / dự đoán.

Mặc định biểu đồ chỉ hiển thị inline trong notebook. Đặt
``PLOT["save_plot"]`` thành một tiền tố (ví dụ ``"save/apple"``) nếu muốn
ghi PNG; khi đó thư mục được tạo tự động.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from lstnet_config import PLOT


def _is_interactive_backend():
    backend = matplotlib.get_backend().lower()
    return any(
        key in backend
        for key in ("tk", "qt", "gtk", "wx", "nbagg", "webagg", "ipympl", "inline")
    )


def _finish(fig, save_path=None):
    if save_path is not None:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(save_path, dpi=PLOT.get("dpi", 120), bbox_inches="tight")
    if PLOT.get("show", True) and _is_interactive_backend():
        plt.show()
    plt.close(fig)


def _mean_curve(curves):
    max_len = max(len(curve) for curve in curves)
    padded = np.full((len(curves), max_len), np.nan)
    for i, curve in enumerate(curves):
        padded[i, :len(curve)] = curve
    return np.nanmean(padded, axis=0)


def PlotHistory(histories, seeds=None, save_path=None):
    """Vẽ loss và val_loss của nhiều seed trên cùng một hình."""
    if not histories:
        return

    if seeds is None:
        seeds = list(range(1, len(histories) + 1))

    fig, axes = plt.subplots(
        1, 2, figsize=PLOT.get("figsize", (12, 4.5)), sharex=False
    )

    for seed, history in zip(seeds, histories):
        epochs = range(1, len(history["loss"]) + 1)
        axes[0].plot(epochs, history["loss"], alpha=0.35, label="seed %s" % seed)
        axes[1].plot(
            epochs, history["val_loss"], alpha=0.35, label="seed %s" % seed
        )

    train_mean = _mean_curve([h["loss"] for h in histories])
    val_mean = _mean_curve([h["val_loss"] for h in histories])
    axes[0].plot(
        range(1, len(train_mean) + 1), train_mean,
        color="black", linewidth=2, label="trung bình",
    )
    axes[1].plot(
        range(1, len(val_mean) + 1), val_mean,
        color="black", linewidth=2, label="trung bình",
    )

    axes[0].set_title("Train loss (MSE, delta đã scale)")
    axes[1].set_title("Validation loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle("Lịch sử huấn luyện LSTNet")
    fig.tight_layout()

    _finish(fig, save_path)


def PlotPrediction(dates, y_true, y_pred, y_naive=None, title=None, save_path=None):
    """So sánh ``nextPrice`` thực tế với dự đoán (và Naive nếu có)."""
    fig, axis = plt.subplots(
        figsize=PLOT.get("prediction_figsize", (14, 5))
    )

    axis.plot(dates, y_true, color="black", linewidth=1.3, label="nextPrice thực tế")
    axis.plot(dates, y_pred, color="tab:red", linewidth=1.1, label="LSTNet dự đoán")
    if y_naive is not None:
        axis.plot(
            dates, y_naive, color="tab:blue", linewidth=0.9,
            linestyle="--", label="Naive (nextPrice = Price)",
        )

    axis.set_title(title or "LSTNet — nextPrice thực tế và dự đoán")
    axis.set_xlabel("Ngày")
    axis.set_ylabel("Giá (USD)")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    _finish(fig, save_path)
