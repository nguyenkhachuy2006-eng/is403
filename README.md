# Dự báo giá cổ phiếu Apple (AAPL) bằng Machine Learning & Mạng Nơ-ron

Tài liệu này dùng để **thống nhất dữ liệu, quy ước và cách đánh giá** cho toàn bộ nhóm khi chạy các mô hình. Mọi mô hình phải tuân theo các quy ước dưới đây để kết quả **so sánh được công bằng với nhau**.

---

## 1. Mục tiêu đồ án

- **Target của đồ án là cột `nextPrice`**: giá đóng cửa của ngày kế tiếp (t+1). Mọi mô hình đều dự đoán cột này, dựa trên thông tin của ngày t (và các ngày trước đó, với mô hình chuỗi).
- **Cách dự đoán đã thống nhất:** mô hình học **`delta = nextPrice - Price`** (mức thay đổi giá từ hôm nay sang ngày mai), sau đó cộng lại `nextPrice_dự_đoán = Price + delta_dự_đoán`. Metric vẫn tính trên giá USD. Lý do và chi tiết xem `Phan_cong_mo_hinh.md`, mục 4, và mục 4.2 bên dưới.
- **Bài báo tham chiếu:** Jin & Yi (ICBBEM 2022), so sánh OLS, Random Forest, XGBoost, LightGBM, LSTM, GRU trên cổ phiếu LONGi. Nhóm mở rộng sang cổ phiếu AAPL với bộ mô hình khác.
- So sánh hiệu quả giữa 5 mô hình cơ sở thuộc 3 nhóm phương pháp, và **Weighted Average Ensemble** kết hợp chúng:
  - Nhóm **Thống kê / Tuyến tính**: ARIMA, Ridge Regression.
  - Nhóm **Học máy**: LightGBM.
  - Nhóm **Học sâu**: LSTNet, Transformer.
- Bài toán chính: **hồi quy** (regression) trên **dữ liệu chuỗi thời gian** (time series). Kèm theo là bài toán **phân loại tăng/giảm** (xem mục 6.2).
- Danh sách mô hình và gợi ý phân công: xem `Phan_cong_mo_hinh.md`.

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

1. **`Change_Pct` và `Return_1` gần như giống nhau** (chỉ lệch do làm tròn, tối đa ≈ 0,0009). Khi dùng mô hình nhạy với đa cộng tuyến (ví dụ Ridge Regression), nên **chỉ giữ một trong hai**.
2. **Giá không dừng (non-stationary)**: giá tăng từ ~10 lên ~340 nên các mô hình dễ học "xu hướng tăng" thay vì quy luật thật. Các thang đo của cột cũng chênh lệch rất lớn (`Volume` ~10⁸, `Return_1` ~10⁻²) nên **bắt buộc chuẩn hoá** khi dùng mạng nơ-ron (xem mục 4).
3. **Đa số feature đều là thông tin của ngày t**, còn `nextPrice` là ngày t+1. Tuyệt đối không dùng thông tin từ ngày t+1 trở đi làm feature.

### 2.4. Chỉ báo kỹ thuật dự kiến (chưa có trong file)

Theo slide, input của nhóm gồm OHLCV kết hợp **RSI, MACD, MA và lợi suất trễ**. Các cột này **chưa có** trong `Apple_Stock_ML_ready.csv`, cần tạo thêm nếu nhóm quyết định dùng:

| Nhóm | Gợi ý tham số chuẩn |
|---|---|
| MA (trung bình động của `Price`) | 5, 10, 20 ngày |
| RSI | 14 ngày |
| MACD | EMA 12 − EMA 26, đường tín hiệu 9 ngày |
| Lợi suất trễ | `Return_1` của các ngày t-1, t-2, t-3, ... |

Mọi chỉ báo chỉ được tính từ dữ liệu **đến hết ngày t**, không dùng dữ liệu từ ngày t+1 trở đi. Các dòng đầu bị NaN do cần dữ liệu quá khứ sẽ bị bỏ. ARIMA là mô hình đơn biến nên không dùng các feature này.

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

- **Tập Test chỉ được dùng cho đánh giá cuối cùng.** Không dùng Test để chỉnh siêu tham số, chọn mô hình hay tính trọng số Ensemble.
- **Mọi mô hình dùng chung cách chia trong Train:** 90% đầu là `train_fit` (để huấn luyện), 10% cuối là `train_val` (làm validation, 322 dòng từ 01/04/2022). Việc dùng chung giúp Weighted Average Ensemble tính trọng số công bằng trên `train_val`.
- **Ridge, LightGBM:** chỉnh siêu tham số bằng **`TimeSeriesSplit`** (walk-forward) trên `train_fit`. Không dùng `KFold` thông thường.
- **LSTNet, Transformer:** dùng `train_val` để theo dõi `val_loss` và EarlyStopping (không shuffle).
- **ARIMA:** chọn bậc `(p, q)` bằng AIC trên `train_fit`.

```python
i_val = int(len(train) * 0.90)
train_fit = train.iloc[:i_val]   # 2.894 dòng, dùng để huấn luyện
train_val = train.iloc[i_val:]   # 322 dòng (từ 01/04/2022), dùng làm validation
```

---

## 4. Quy ước tiền xử lý

| Nội dung | Quy ước thống nhất |
|---|---|
| Chuẩn hoá | `StandardScaler` hoặc `MinMaxScaler`, **chỉ `fit` trên tập Train**, sau đó `transform` cho Test |
| Biến mà mô hình học | **`delta = nextPrice - Price`** (xem mục 4.2), không học trực tiếp `nextPrice` |
| Chuẩn hoá `delta` | Nếu LSTNet / Transformer train trên `delta` đã scale, phải **`inverse_transform`**, cộng lại với `Price`, rồi mới tính metric |
| Rò rỉ dữ liệu (data leakage) | Cấm `fit` scaler, chọn feature hay tính thống kê trên toàn bộ dataset |
| Bỏ cột | Bỏ `Date` và `nextPrice` khỏi ma trận feature `X` |
| Tập feature cơ sở | `Price, Open, High, Low, Volume, Return_1, High_Low_Range, Open_Close_Change` |
| Tập feature mở rộng | Cơ sở + `Year, Month, Day, DayOfWeek` (báo cáo riêng nếu dùng) |
| Tập feature kỹ thuật (dự kiến) | Cơ sở + RSI, MACD, MA, lợi suất trễ (xem mục 2.4, báo cáo riêng nếu dùng) |
| ARIMA | Chỉ dùng chuỗi `Price` (đơn biến), không cần ma trận feature |

```python
from sklearn.preprocessing import StandardScaler

features = ['Price', 'Open', 'High', 'Low', 'Volume',
            'Return_1', 'High_Low_Range', 'Open_Close_Change']

scaler = StandardScaler()
X_train = scaler.fit_transform(train[features])   # chỉ fit trên train
X_test  = scaler.transform(test[features])
```

### 4.1. Dữ liệu dạng cửa sổ cho mô hình chuỗi (LSTNet / Transformer)

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

Giá trong tập Test (đến 341,07 USD) vượt xa giá trong tập Train (tối đa 193,97 USD), nên mô hình dự đoán thẳng `nextPrice` sẽ rất sai, nhất là LightGBM. Vì vậy **cả nhóm cho mô hình học `delta = nextPrice - Price`** rồi cộng lại thành giá.

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
- ARIMA dùng `d = 1` (sai phân bậc 1) nên đã tương đương với việc học trên `delta`, và dự báo trả ra trực tiếp là giá `nextPrice`.

---

## 5. Mô hình sẽ chạy

| Nhóm | Mô hình |
|---|---|
| Baseline | Naive (dự đoán `nextPrice = Price` hôm nay) |
| Thống kê / Tuyến tính | ARIMA, Ridge Regression |
| Học máy | LightGBM |
| Học sâu | LSTNet, Transformer |
| Kết hợp | Weighted Average Ensemble của 5 mô hình cơ sở (xem `Phan_cong_mo_hinh.md`, mục 5) |

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

### 6.2. Bài toán phân loại tăng/giảm

Theo slide, đầu ra thứ hai là **xu hướng biến động tăng/giảm** của ngày kế tiếp:

- Nhãn: `Direction = 1` nếu `nextPrice > Price`, ngược lại `Direction = 0`.
- Dự đoán: `Direction_pred = 1` nếu `delta_pred > 0`, ngược lại `0` (suy ra từ dấu của `delta` dự đoán, không train bộ phân loại riêng).
- Metric: **Accuracy** (tương đương Directional Accuracy ở trên) và **F1-score**.

---

## 7. Quy ước thực nghiệm để tái lập kết quả

- **Random seed cố định = 42** cho `numpy`, `random`, `sklearn`, `tensorflow` / `torch`.
- Chạy LSTNet và Transformer **ít nhất 3 lần với seed khác nhau** và báo cáo trung bình ± độ lệch chuẩn.
- LSTNet và Transformer dùng **EarlyStopping** theo `val_loss` (gợi ý `patience = 10–20`), lưu lại epoch tốt nhất.
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
| ARIMA | Price | – | | | | | | – | `(p,1,q)` = ... |
| Ridge | Cơ sở | – | | | | | | 42 | Target: delta |
| LightGBM | Cơ sở | – | | | | | | 42 | Target: delta |
| LSTNet | Cơ sở | 30 | | | | | | 42 | Target: delta |
| Transformer | Cơ sở | 30 | | | | | | 42 | Target: delta |
| Weighted Average Ensemble | – | – | | | | | | – | Trọng số từ `train_val` |

---

## 9. Thư viện cần cài

```
pandas
numpy
scikit-learn
matplotlib
seaborn
statsmodels   # ARIMA
lightgbm
tensorflow     # hoặc torch, thống nhất trong nhóm dùng một loại (cho LSTNet, Transformer)
jupyter
```

---

## 10. Checklist trước khi nộp kết quả

- [ ] Chia tập theo thời gian 80/20, **không shuffle**
- [ ] Scaler chỉ `fit` trên Train
- [ ] Không có feature nào chứa thông tin từ ngày t+1 trở đi
- [ ] Mô hình học `delta = nextPrice - Price`, đã cộng lại thành giá trước khi tính metric
- [ ] Có baseline Naive để so sánh
- [ ] Dùng chung `train_fit` / `train_val`, trọng số Ensemble chỉ tính từ `train_val`
- [ ] Metric tính trên giá gốc (USD), đã `inverse_transform`
- [ ] Tập Test chỉ dùng cho đánh giá cuối cùng
- [ ] Đã cố định seed và ghi lại siêu tham số
- [ ] Kết quả điền đúng mẫu bảng ở mục 8
