# Dự báo giá cổ phiếu Apple (AAPL) bằng Machine Learning & Mạng Nơ-ron

Tài liệu này dùng để **thống nhất dữ liệu, quy ước và cách đánh giá** cho toàn bộ nhóm khi chạy các mô hình. Mọi mô hình phải tuân theo các quy ước dưới đây để kết quả **so sánh được công bằng với nhau**.

---

## 1. Mục tiêu đồ án

- **Target của đồ án là cột `nextPrice`**: giá đóng cửa của ngày kế tiếp (t+1). Mọi mô hình đều dự đoán cột này, dựa trên thông tin của ngày t (và các ngày trước đó, với mô hình chuỗi).
- **Cách dự đoán:** mô hình học **`delta = nextPrice - Price`** (mức thay đổi giá từ hôm nay sang ngày mai), sau đó cộng lại `nextPrice_dự_đoán = Price + delta_dự_đoán`. Metric vẫn tính trên giá USD. Lý do và chi tiết xem `Phan_cong_mo_hinh.md`, mục 4, và mục 4.2 bên dưới.
- So sánh hiệu quả giữa:
  - Nhóm **Machine Learning** truyền thống.
  - Nhóm **Mạng nơ-ron** (MLP, RNN/LSTM/GRU, CNN 1D, ...).
- Bài toán: **hồi quy** (regression) trên **dữ liệu chuỗi thời gian** (time series).

---

## 2. Dữ liệu

| Mục | Giá trị |
|---|---|
| Nguồn gốc | `Apple_Stock_Price_History.csv` (lịch sử giá cổ phiếu Apple theo ngày) |
| File dùng để train | `Apple_Stock_ML_ready.csv` |
| Số dòng / số cột | 4.020 dòng × 15 cột |
| Khoảng thời gian | 04/10/2010 → 28/09/2026 |
| Tần suất | Theo ngày giao dịch (không có thứ Bảy, Chủ Nhật, ngày lễ) |
| Giá trị thiếu | Không có (đã loại 2 dòng NaN ở đầu và cuối) |
| Thứ tự dòng | **Cũ → mới** (đã sắp xếp lại so với file gốc) |

### 2.1. Mô tả các cột

| Cột | Kiểu | Ý nghĩa | Vai trò |
|---|---|---|---|
| `Date` | datetime | Ngày giao dịch | Chỉ dùng để sắp xếp / chia tập, **không đưa vào mô hình** |
| `Year` | int | Năm | Feature (tuỳ chọn) |
| `Month` | int | Tháng (1–12) | Feature (tuỳ chọn) |
| `Day` | int | Ngày trong tháng | Feature (tuỳ chọn) |
| `DayOfWeek` | int | Thứ trong tuần (0 = Thứ Hai, 4 = Thứ Sáu) | Feature (tuỳ chọn) |
| `Price` | float | Giá đóng cửa ngày t | Feature |
| `Open` | float | Giá mở cửa | Feature |
| `High` | float | Giá cao nhất trong ngày | Feature |
| `Low` | float | Giá thấp nhất trong ngày | Feature |
| `Volume` | float | Khối lượng giao dịch (số cổ phiếu) | Feature |
| `Change_Pct` | float | % thay đổi so với ngày trước, dạng thập phân (-0.0199 = -1,99%) | Feature (xem lưu ý 2.3) |
| `Return_1` | float | Lợi suất 1 ngày = `Price_t / Price_{t-1} - 1` | Feature |
| `High_Low_Range` | float | `High - Low` (biên độ dao động trong ngày) | Feature |
| `Open_Close_Change` | float | `Price - Open` (chênh lệch đóng – mở) | Feature |
| **`nextPrice`** | float | **`Price` của ngày t+1** | **TARGET – biến cần dự đoán** |

> `Price` ở đây là **giá đóng cửa** (Close).
>
> 🎯 **Target thống nhất của cả nhóm: `nextPrice`** (trước đây đặt tên là `Target`). Chỉ dự đoán `nextPrice`, không dùng nó làm feature.

### 2.2. Thống kê nhanh

| Cột | Min | Trung bình | Max |
|---|---|---|---|
| `Price` | 9,95 | 92,84 | 341,07 |
| `Volume` | 17,91 triệu | 191,03 triệu | 1,88 tỷ |
| `Return_1` | -12,87% | +0,10% | +15,33% |
| `High_Low_Range` | 0,05 | 1,94 | 28,72 |
| `Open_Close_Change` | -14,28 | 0,08 | 26,90 |

### 2.3. Lưu ý quan trọng về dữ liệu

1. **`Change_Pct` và `Return_1` gần như giống nhau** (chỉ lệch do làm tròn, tối đa ≈ 0,0009). Khi dùng mô hình nhạy với đa cộng tuyến (ví dụ Linear Regression), nên **chỉ giữ một trong hai**.
2. **Giá không dừng (non-stationary)**: giá tăng từ ~10 lên ~340 nên các mô hình dễ học "xu hướng tăng" thay vì quy luật thật. Các thang đo của cột cũng chênh lệch rất lớn (`Volume` ~10⁸, `Return_1` ~10⁻²) nên **bắt buộc chuẩn hoá** khi dùng mạng nơ-ron (xem mục 4).
3. **Đa số feature đều là thông tin của ngày t**, còn `nextPrice` là ngày t+1. Tuyệt đối không dùng thông tin từ ngày t+1 trở đi làm feature.

---

## 3. Quy ước chia dữ liệu (BẮT BUỘC)

**Không được shuffle ngẫu nhiên** dữ liệu chuỗi thời gian. Chia theo trình tự thời gian, tỉ lệ **80 / 20**:

| Tập | Tỉ lệ | Số dòng | Khoảng thời gian |
|---|---|---|---|
| Train | 80% | 3.216 | 04/10/2010 → 14/07/2023 |
| Test | 20% | 804 | 17/07/2023 → 28/09/2026 |

```python
n = len(data)
i_train = int(n * 0.80)

train = data.iloc[:i_train]
test  = data.iloc[i_train:]
```

Quy tắc đi kèm:

- **Tập Test chỉ được dùng cho đánh giá cuối cùng.** Không dùng Test để chỉnh siêu tham số hay chọn mô hình.
- Với mô hình **Machine Learning**: chỉnh siêu tham số bằng **`TimeSeriesSplit`** (walk-forward) trên tập Train. Không dùng `KFold` thông thường.
- Với **mạng nơ-ron**: lấy **10% cuối của tập Train** làm tập validation để theo dõi `val_loss` và dùng EarlyStopping (không shuffle). Phần này nằm trong Train, không lấy từ Test.

```python
i_val = int(len(train) * 0.90)
train_fit = train.iloc[:i_val]   # 2.894 dòng, dùng để train
train_val = train.iloc[i_val:]   # 322 dòng (từ 01/04/2022), dùng cho EarlyStopping
```

---

## 4. Quy ước tiền xử lý

| Nội dung | Quy ước thống nhất |
|---|---|
| Chuẩn hoá | `StandardScaler` hoặc `MinMaxScaler`, **chỉ `fit` trên tập Train**, sau đó `transform` cho Test |
| Biến mà mô hình học | **`delta = nextPrice - Price`** (xem mục 4.2), không học trực tiếp `nextPrice` |
| Chuẩn hoá `delta` | Nếu mạng nơ-ron train trên `delta` đã scale, phải **`inverse_transform`**, cộng lại với `Price`, rồi mới tính metric |
| Rò rỉ dữ liệu (data leakage) | Cấm `fit` scaler, chọn feature hay tính thống kê trên toàn bộ dataset |
| Bỏ cột | Bỏ `Date` và `nextPrice` khỏi ma trận feature `X` |
| Tập feature cơ sở | `Price, Open, High, Low, Volume, Return_1, High_Low_Range, Open_Close_Change` |
| Tập feature mở rộng | Cơ sở + `Year, Month, Day, DayOfWeek` (báo cáo riêng nếu dùng) |

```python
from sklearn.preprocessing import StandardScaler

features = ['Price', 'Open', 'High', 'Low', 'Volume',
            'Return_1', 'High_Low_Range', 'Open_Close_Change']

scaler = StandardScaler()
X_train = scaler.fit_transform(train[features])   # chỉ fit trên train
X_test  = scaler.transform(test[features])
```

### 4.1. Dữ liệu dạng cửa sổ cho mô hình chuỗi (LSTM / GRU / CNN 1D)

- Độ dài cửa sổ mặc định: **`WINDOW = 30`** ngày. Có thể thử thêm 10 và 60, nhưng phải ghi lại trong báo cáo.
- Đầu vào có dạng `(số mẫu, WINDOW, số feature)`.
- Mỗi mẫu dùng `WINDOW` ngày liên tiếp `[t-WINDOW+1 … t]` để dự đoán `delta` của ngày t (`nextPrice = Price_t + delta`).
- Cửa sổ chỉ được tạo **trong từng tập** hoặc bằng cách ghép phần đuôi của tập trước để không mất mẫu đầu, nhưng **không được để cửa sổ chứa dữ liệu tương lai**.

```python
import numpy as np

def make_windows(X, y, window=30):
    Xs, ys = [], []
    for i in range(window - 1, len(X)):
        Xs.append(X[i - window + 1 : i + 1])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)
```

### 4.2. Biến mà mô hình học (ĐÃ CHỐT)

Giá trong tập Test (đến 341,07 USD) vượt xa giá trong tập Train (tối đa 193,97 USD), nên mô hình dự đoán thẳng `nextPrice` sẽ rất sai, nhất là Random Forest, XGBoost và LightGBM. Vì vậy **cả nhóm cho mô hình học `delta = nextPrice - Price`** rồi cộng lại thành giá.

```python
y_train_delta = train['nextPrice'] - train['Price']     # biến mô hình học
model.fit(X_train, y_train_delta)

delta_pred = model.predict(X_test)
price_pred = test['Price'].values + delta_pred          # dự đoán nextPrice

results = evaluate(test['nextPrice'].values, price_pred, test['Price'].values)
```

- `nextPrice` vẫn là **target chính thức**, metric vẫn tính trên **giá USD**.
- `delta` và `nextPrice` **không** đưa vào ma trận feature `X`.
- Phân tích đầy đủ: xem `Phan_cong_mo_hinh.md`, mục 4.

---

## 5. Mô hình đề xuất

| Nhóm | Mô hình |
|---|---|
| Baseline | Naive (dự đoán `nextPrice = Price` hôm nay), Linear Regression |
| Machine Learning | Ridge / Lasso, SVR, Random Forest, Gradient Boosting / XGBoost / LightGBM |
| Mạng nơ-ron | MLP, LSTM, GRU, CNN 1D (có thể mở rộng thêm CNN-LSTM, Transformer) |

**Bắt buộc có baseline Naive** để làm mốc. Giá cổ phiếu hôm nay thường rất gần giá ngày mai, nên một mô hình phức tạp mà không thắng được Naive thì không có giá trị thực tế.

---

## 6. Metric đánh giá thống nhất

Tất cả metric được tính trên **giá thực (USD) sau khi đã inverse_transform**, không tính trên dữ liệu đã scale.

| Metric | Ý nghĩa | Bắt buộc |
|---|---|---|
| **MAE** | Sai số tuyệt đối trung bình (USD) | Có |
| **RMSE** | Căn bậc hai sai số bình phương (phạt nặng sai số lớn) | Có |
| **MAPE (%)** | Sai số phần trăm trung bình | Có |
| **R²** | Mức độ giải thích phương sai | Có (chỉ mang tính tham khảo vì giá có xu hướng) |
| **Directional Accuracy (%)** | Tỉ lệ dự đoán đúng hướng tăng / giảm so với hôm nay | Khuyến khích |

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate(y_true, y_pred, y_today):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2   = r2_score(y_true, y_pred)
    da   = np.mean(np.sign(y_pred - y_today) == np.sign(y_true - y_today)) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2, 'DirAcc': da}
```

> `y_true` là `nextPrice` thực tế, `y_pred` là `Price + delta_dự_đoán` (đã cộng lại thành giá), `y_today` là cột `Price` (giá hôm nay) tương ứng với từng mẫu.

### 6.1. Mốc tham chiếu (Naive baseline, trên tập Test 80/20)

| Mô hình | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| Naive (`nextPrice = Price`) | ≈ 2,62 | ≈ 3,84 | ≈ 1,14 |

Mô hình nào muốn chứng minh hiệu quả thì cần **thấp hơn** mốc này.

---

## 7. Quy ước thực nghiệm để tái lập kết quả

- **Random seed cố định = 42** cho `numpy`, `random`, `sklearn`, `tensorflow` / `torch`.
- Chạy mô hình nơ-ron **ít nhất 3 lần với seed khác nhau** và báo cáo trung bình ± độ lệch chuẩn.
- Mạng nơ-ron dùng **EarlyStopping** theo `val_loss` (gợi ý `patience = 10–20`), lưu lại epoch tốt nhất.
- Loss mặc định: **MSE** (có thể thử Huber). Optimizer mặc định: **Adam**, `lr = 1e-3`.
- Ghi lại toàn bộ siêu tham số, số lượng tham số mô hình và thời gian train của mỗi lần chạy.

```python
import random, numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# tensorflow: tf.random.set_seed(SEED)
# torch:      torch.manual_seed(SEED)
```

---

## 8. Mẫu bảng kết quả (mọi thành viên điền theo đúng định dạng này)

Kết quả trên **tập Test**, đơn vị USD (trừ MAPE, R², DirAcc):

| Mô hình | Feature | Window | MAE | RMSE | MAPE (%) | R² | DirAcc (%) | Seed | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|
| Naive | – | – | 2,62 | 3,84 | 1,14 | | | – | Baseline (delta = 0) |
| Linear Regression | Cơ sở | – | | | | | | 42 | |
| Random Forest | Cơ sở | – | | | | | | 42 | |
| MLP | Cơ sở | – | | | | | | 42 | |
| LSTM | Cơ sở | 30 | | | | | | 42 | |
| GRU | Cơ sở | 30 | | | | | | 42 | |

---

## 9. Thư viện cần cài

```
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
lightgbm
tensorflow     # hoặc torch, thống nhất trong nhóm dùng một loại
jupyter
```

---

## 10. Checklist trước khi nộp kết quả

- [ ] Chia tập theo thời gian 80/20, **không shuffle**
- [ ] Scaler chỉ `fit` trên Train
- [ ] Không có feature nào chứa thông tin từ ngày t+1 trở đi
- [ ] Mô hình học `delta = nextPrice - Price`, đã cộng lại thành giá trước khi tính metric
- [ ] Có baseline Naive để so sánh
- [ ] Metric tính trên giá gốc (USD), đã `inverse_transform`
- [ ] Tập Test chỉ dùng cho đánh giá cuối cùng
- [ ] Đã cố định seed và ghi lại siêu tham số
- [ ] Kết quả điền đúng mẫu bảng ở mục 8
