"""Đọc dữ liệu, chia tập và tạo cửa sổ cho LSTNet.

Quy ước của nhóm được áp dụng ở đây:

- Target chính thức: ``nextPrice``.
- Biến mô hình học: ``delta = nextPrice - Price``.
- Chia theo thời gian: 80% Train / 20% Test. Trong Train: 90% ``train_fit``
  và 10% ``train_val``.
- Scaler feature và scaler ``delta`` chỉ fit trên ``train_fit``.
- Cửa sổ ``window`` ngày: mẫu kết thúc ở ngày ``t`` dùng các ngày
  ``t-window+1 .. t`` để dự đoán ``delta_t``; giá dự đoán là
  ``nextPrice_t = Price_t + delta_t``.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _resolve_column(df, name):
    """Tìm tên cột, không phân biệt hoa thường (NextPrice / nextPrice)."""
    if name in df.columns:
        return name
    for column in df.columns:
        if column.lower() == name.lower():
            return column
    raise KeyError(
        "Không tìm thấy cột %r. Các cột hiện có: %s" % (name, list(df.columns))
    )


class WindowSet(object):
    """Tập mẫu đã tạo cửa sổ.

    - ``X``: (n, window, n_features) — feature đã scale.
    - ``AR``: (n, highway, 1) — lịch sử ``delta`` đã scale, chỉ dùng quá khứ.
    - ``y``: (n,) — ``delta`` đã scale của ngày cuối cửa sổ.
    - ``dates``, ``price_today``, ``next_price``: mốc thời gian và giá gốc.
    """

    def __init__(self, X, AR, y, dates, price_today, next_price):
        self.X = X
        self.AR = AR
        self.y = y
        self.dates = dates
        self.price_today = price_today
        self.next_price = next_price

    def __len__(self):
        return len(self.y)


class StockData(object):
    """Dữ liệu Apple đã chia tập, chuẩn hoá và tạo cửa sổ."""

    def __init__(self, init):
        self.init = init

        df = pd.read_csv(init.data)

        """Kiểm tra feature và cột, sắp xếp theo thời gian."""
        self.date_col = _resolve_column(df, init.date_col)
        self.price_col = _resolve_column(df, init.price_col)
        self.target_col = _resolve_column(df, init.target_col)

        missing = [column for column in init.features if column not in df.columns]
        if missing:
            raise KeyError(
                "Thiếu feature %s trong %s. Các cột hiện có: %s"
                % (missing, init.data, list(df.columns))
            )

        """Chuyển cột ngày về datetime, sắp xếp theo ngày tăng dần và bỏ NaN."""
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.sort_values(self.date_col).reset_index(drop=True)
        df = df.dropna(subset=init.features + [self.price_col, self.target_col])
        df = df.reset_index(drop=True)
        self.df = df

        """Kiểm tra đủ dữ liệu cho cửa sổ và chia tập theo thời gian."""
        n = len(df)
        if n <= init.window + 1:
            raise ValueError(
                "Không đủ dữ liệu: %d dòng, window=%d" % (n, init.window)
            )

        """Chia tập theo thời gian"""
        self.i_train = int(n * init.train_ratio)
        self.i_val = int(self.i_train * init.fit_ratio)
        self.train_frame = df.iloc[:self.i_train]
        self.train_fit_frame = df.iloc[:self.i_val]
        self.train_val_frame = df.iloc[self.i_val:self.i_train]
        self.test_frame = df.iloc[self.i_train:]

        # --- Giá và delta gốc ---
        self.price_all = df[self.price_col].to_numpy(dtype=float)
        self.next_all = df[self.target_col].to_numpy(dtype=float)
        self.delta_all_raw = self.next_all - self.price_all
        self.dates = df[self.date_col].to_numpy()

        # --- Scaler chỉ fit trên train_fit ---
        scaler_cls = MinMaxScaler if init.scaler == "minmax" else StandardScaler
        self.feature_scaler = scaler_cls().fit(
            self.train_fit_frame[init.features].to_numpy(dtype=float)
        )
        self.delta_scaler = StandardScaler().fit(
            self.delta_all_raw[:self.i_val].reshape(-1, 1)
        )

        self.X_all = self.feature_scaler.transform(
            df[init.features].to_numpy(dtype=float)
        )
        self.delta_all = self.delta_scaler.transform(
            self.delta_all_raw.reshape(-1, 1)
        ).ravel()

        # --- Tạo cửa sổ ---
        self.fit = self._make_set(init.window - 1, self.i_val)
        self.val = self._make_set(self.i_val, self.i_train)
        self.test = self._make_set(self.i_train, n)

        # Kiểm tra nhanh để tránh lỗi cấu hình
        if len(self.fit) == 0 or len(self.val) == 0 or len(self.test) == 0:
            raise ValueError("Một trong các tập fit/val/test không có mẫu nào")

    def _make_set(self, start, end):
        """Tạo mẫu cho các dòng [start, end).

        Cửa sổ được phép lấy đuôi của tập trước (dữ liệu quá khứ) để không
        mất mẫu đầu của val/test, nhưng không bao giờ chứa dữ liệu tương lai.
        """
        window = self.init.window
        highway = self.init.highway
        indices = np.arange(start, end)

        X = np.stack([self.X_all[i - window + 1:i + 1] for i in indices])

        if highway > 0:
            AR = np.stack([
                self.delta_all[i - highway:i].reshape(-1, 1)
                for i in indices
            ])
        else:
            AR = np.zeros((len(indices), 0, 1), dtype=float)

        return WindowSet(
            X=X,
            AR=AR,
            y=self.delta_all[indices],
            dates=self.dates[indices],
            price_today=self.price_all[indices],
            next_price=self.next_all[indices],
        )

    def inverse_delta(self, values):
        """Đưa ``delta`` đã scale về USD."""
        values = np.asarray(values, dtype=float).reshape(-1, 1)
        return self.delta_scaler.inverse_transform(values).ravel()

    def prediction_to_price(self, delta_scaled, window_set):
        """Ghép ``delta`` dự đoán với giá hôm nay để ra ``nextPrice``."""
        delta = self.inverse_delta(delta_scaled)
        return window_set.price_today + delta

    def split_summary(self):
        """Bảng tóm tắt các tập để in trong notebook."""
        rows = []
        for name, frame in (
            ("train_fit", self.train_fit_frame),
            ("train_val", self.train_val_frame),
            ("test", self.test_frame),
        ):
            rows.append({
                "Tập": name,
                "Số dòng": len(frame),
                "Từ ngày": frame[self.date_col].iloc[0].date(),
                "Đến ngày": frame[self.date_col].iloc[-1].date(),
            })
        return pd.DataFrame(rows)
