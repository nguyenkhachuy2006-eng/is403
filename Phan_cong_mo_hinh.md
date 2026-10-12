# Danh sách mô hình & phân công (tham khảo)

File này bổ sung cho `README.md` và khớp với slide giới thiệu đề tài (Nhóm 2 - IS403.R11). Mọi quy ước về **dữ liệu, chia tập, tiền xử lý, metric, seed** vẫn theo `README.md`. File này nói về **mô hình nào sẽ được chạy**, **gợi ý ai làm gì**, **cách xây dựng Weighted Average Ensemble** và **một vấn đề quan trọng về dữ liệu đã được thống nhất cách xử lý** (mục 4).

> ✅ **ĐÃ CHỐT:** mọi mô hình học **`delta = nextPrice - Price`** (mức thay đổi giá từ hôm nay sang ngày mai), sau đó cộng lại `nextPrice_dự_đoán = Price + delta_dự_đoán` để tính metric trên giá USD. Chi tiết ở mục 4.

> ⚠️ **Phân công dưới đây chỉ mang tính tham khảo.** Với mạng nơ-ron (LSTNet, Transformer), mỗi người hoàn toàn có thể kết hợp nhiều kiến trúc (ví dụ thêm CNN, GRU, Attention, ...) và không bắt buộc phải theo đúng danh sách. Điều kiện duy nhất: vẫn tuân thủ quy ước chung trong `README.md` và ghi rõ kiến trúc đã dùng ở cột *Ghi chú* của bảng kết quả.

---

## 1. Các mô hình sẽ chạy

Theo slide, nhóm chạy **5 mô hình cơ sở** thuộc 3 nhóm phương pháp, sau đó kết hợp chúng bằng **Weighted Average Ensemble**.

| Nhóm phương pháp | Mô hình | Dạng đầu vào |
|---|---|---|
| Thống kê / Tuyến tính | **ARIMA** | Chuỗi `Price` đơn biến (không dùng các feature khác) |
| Thống kê / Tuyến tính | **Ridge Regression** | Bảng 2D `(mẫu, feature)` |
| Học máy | **LightGBM** | Bảng 2D `(mẫu, feature)` |
| Học sâu | **LSTNet** | Cửa sổ 3D `(mẫu, WINDOW, feature)` |
| Học sâu | **Transformer** | Cửa sổ 3D `(mẫu, WINDOW, feature)` |
| Kết hợp | **Weighted Average Ensemble** | Dự đoán của 5 mô hình trên |
| Mốc tham chiếu | Naive (`nextPrice = Price`) | – |

---

## 2. Phân công gợi ý (5 thành viên)

Mỗi người phụ trách **một mô hình cơ sở**:

| Thành viên | Mô hình |
|---|---|
| A | ARIMA |
| B | Ridge Regression |
| C | LightGBM |
| D | LSTNet |
| E | Transformer |

- **Weighted Average Ensemble** làm sau khi cả 5 mô hình có dự đoán. Người phụ trách chưa phân công (xem mục 7).
- Naive baseline dùng **chung một hàm** trong code chung nên không cần giao riêng.

---

## 3. Gợi ý cho từng mô hình

### ARIMA

- Mô hình đơn biến trên chuỗi `Price`, **không dùng** `Open, High, Low, Volume, ...`.
- Dùng **sai phân bậc 1 (`d = 1`)**, tức `ARIMA(p, 1, q)`. Khi đó mô hình học trên chênh lệch giá liên tiếp, **tương đương với quy ước `delta` đã chốt** (mục 4), và kết quả dự báo trả ra trực tiếp là giá `nextPrice`.
- Chọn `(p, q)` bằng AIC (hoặc BIC) trên `train_fit`, chỉ dùng dữ liệu Train.
- **Dự báo từng bước một (one-step-ahead):** mỗi ngày đưa giá thực tế mới vào mô hình rồi dự báo ngày kế tiếp. Không dự báo một mạch cả đoạn Test, vì dự báo nhiều bước của ARIMA sẽ gần như thành một đường thẳng và không cùng bài toán với các mô hình khác.

```python
from statsmodels.tsa.arima.model import ARIMA

res = ARIMA(train_fit['Price'].values, order=(p, 1, q)).fit()

price_pred = []
for price_t in eval_df['Price'].values:      # eval_df = train_val hoặc test, theo đúng thứ tự thời gian
    res = res.append([price_t], refit=False)  # nạp giá thực tế của ngày t
    price_pred.append(res.forecast(1)[0])     # dự báo nextPrice của ngày t
```

> Đoạn code trên chỉ là mẫu tham khảo, người phụ trách cần chạy thử và chỉnh theo phiên bản `statsmodels` đang dùng.

### Ridge Regression

- Chuẩn hoá feature bằng `StandardScaler` (chỉ `fit` trên Train).
- Chọn hệ số điều chuẩn `alpha` bằng **`TimeSeriesSplit`** trên `train_fit` (ví dụ thử `alpha` từ 0,01 đến 100 theo thang log).
- Ridge xử lý được các feature tương quan cao như `Price, Open, High, Low`, nhưng vẫn nên bỏ một trong hai cột `Change_Pct` và `Return_1` (gần như trùng nhau).

### LightGBM

- **Bắt buộc học trên `delta`** (xem mục 4), nếu không dự đoán sẽ bị chặn quanh mức giá lớn nhất của Train.
- Siêu tham số nên thử: `num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`, `feature_fraction`.
- Có thể dùng `early_stopping` trên `train_val`.
- Chỉnh siêu tham số bằng **`TimeSeriesSplit`** trên `train_fit`.

### LSTNet

- Kiến trúc gồm Conv1D (bắt đặc trưng cục bộ), GRU, skip-GRU (bắt chu kỳ) và thành phần tự hồi quy (highway).
- Cửa sổ mặc định `WINDOW = 30`.
- Siêu tham số nên thử: `kernel size`, `hidden size`, `skip` (ví dụ 5 ngày tương ứng một tuần giao dịch), cửa sổ highway.

### Transformer

- Dạng encoder dùng self-attention, cần **positional encoding**.
- Cửa sổ mặc định `WINDOW = 30`.
- Siêu tham số nên thử: số head, số lớp encoder, `d_model`, dropout.

### Chung cho LSTNet và Transformer

- Scale `delta` bằng scaler `fit` trên Train, `inverse_transform` rồi cộng lại với `Price` của ngày cuối cửa sổ.
- Dùng EarlyStopping theo `val_loss` trên `train_val`, chạy ít nhất 3 seed như `README.md`.

---

## 4. Vấn đề quan trọng: giá trong tập Test vượt khỏi phạm vi tập Train

Đây là vấn đề **ảnh hưởng đến các mô hình**. Nhóm đã thống nhất cách xử lý ở mục 4.4. Các mục 4.1 đến 4.3 giải thích lý do.

### 4.1. Hiện tượng

Dữ liệu là chuỗi thời gian và giá Apple tăng mạnh theo thời gian, nên khi chia 80/20 theo thứ tự thời gian:

| | Khoảng giá `Price` (USD) |
|---|---|
| Tập Train (04/10/2010 → 14/07/2023) | 9,95 → **193,97** |
| Tập Test (17/07/2023 → 28/09/2026) | 165,00 → **341,07** |

Khoảng **74% số dòng trong tập Test** có `Price` cao hơn mức cao nhất mà mô hình từng thấy trong lúc train. Nói cách khác, đa số ngày Test nằm ở vùng giá mô hình **chưa bao giờ được học**.

### 4.2. Vì sao đây là vấn đề

**Với LightGBM (mô hình dựa trên cây quyết định):**
- Mỗi lá của cây trả về một giá trị lấy từ dữ liệu train. Vì vậy dự đoán **không thể vượt quá** giá trị lớn nhất của target trong tập Train (≈ 194 USD).
- Nếu ta bắt mô hình dự đoán thẳng `nextPrice`, thì khi giá thực tế lên 250, 300, 340 USD, mô hình vẫn chỉ trả về khoảng 190 USD. Sai số sẽ lớn dần theo thời gian, không phải vì mô hình "học dở" mà vì **về mặt cấu trúc nó không thể dự đoán vùng giá đó**.

**Với LSTNet và Transformer:**
- Scaler được `fit` trên Train nên giá Test sau khi scale sẽ vượt ra ngoài khoảng đã quen (ví dụ với `MinMaxScaler`, giá trị > 1). Mạng chưa từng thấy đầu vào ở vùng này nên dự đoán kém ổn định.
- Không hỏng cứng như mô hình cây, nhưng chất lượng thường giảm đáng kể.

**Với Ridge Regression:**
- Mô hình tuyến tính ngoại suy được nên ít bị ảnh hưởng hơn (xem bảng ở mục 4.3).

**Với ARIMA:**
- Dùng `d = 1` nên mô hình đã học trên chênh lệch giá liên tiếp, không bị ảnh hưởng.

### 4.3. Thực nghiệm nhanh để minh hoạ

Thử nghiệm nhanh trên feature cơ sở theo `README.md`, chia 80/20 theo thời gian, chưa tinh chỉnh siêu tham số. Metric tính trên giá USD của tập Test. Vì môi trường thử chưa cài LightGBM nên dùng `HistGradientBoostingRegressor` của scikit-learn làm đại diện (cùng họ gradient boosting trên cây, `random_state=42`).

| Cách làm | MAE | RMSE | Nhận xét |
|---|---|---|---|
| Naive (`nextPrice = Price`) | 2,62 | 3,84 | Mốc tham chiếu |
| Gradient boosting, dự đoán thẳng `nextPrice` | **45,50** | **61,13** | Dự đoán lớn nhất chỉ ≈ 190,9, đúng như phân tích ở trên |
| Gradient boosting, học `delta` rồi cộng lại | 2,86 | 4,08 | Sai số giảm hơn 15 lần |
| Ridge, dự đoán thẳng `nextPrice` | 2,67 | 3,92 | Mô hình tuyến tính ngoại suy được |
| Ridge, học `delta` rồi cộng lại | 2,63 | 3,87 | Cải thiện nhẹ |

Kết luận từ thực nghiệm:
- Mô hình cây dự đoán thẳng `nextPrice` cho kết quả **tệ hơn Naive hàng chục lần**, và nếu không biết nguyên nhân, nhóm sẽ dễ kết luận nhầm là "LightGBM không phù hợp".
- Chỉ cần đổi **cách biểu diễn target** là sai số giảm từ 45,50 xuống 2,86.

### 4.4. Cách xử lý đã chốt: học `delta = nextPrice - Price`

**Cả nhóm dùng cách này cho TẤT CẢ mô hình** để kết quả so sánh được công bằng. Cụ thể:

| Nội dung | Quy ước đã chốt |
|---|---|
| Target chính thức của đồ án | Cột `nextPrice` (không đổi) |
| Biến mà mô hình học | **`delta = nextPrice - Price`** (ARIMA dùng `d = 1` nên tương đương) |
| Mô hình dự đoán | Mức thay đổi giá từ hôm nay sang ngày mai (USD), dấu `+` là giá tăng, dấu `-` là giá giảm |
| Khôi phục về giá | `nextPrice_dự_đoán = Price + delta_dự_đoán` |
| Metric | Tính trên **giá USD** (`nextPrice` thực tế so với `nextPrice_dự_đoán`) |
| Giá `Price` được cộng lại | Giá thực tế của ngày t đã biết. Với mô hình dùng cửa sổ, là `Price` của ngày cuối cùng trong cửa sổ |
| Naive baseline | Tương ứng với việc dự đoán `delta = 0` |

**Ví dụ (số giả định):**

| | Giá trị |
|---|---|
| `Price` hôm nay (đã biết) | 300,00 |
| Mô hình dự đoán `delta` | +1,20 |
| `nextPrice` dự đoán = 300,00 + 1,20 | 301,20 |
| `nextPrice` thực tế | 302,50 |
| Sai số (tính trên giá USD) | 1,30 |

**Quy trình chung cho Ridge, LightGBM, LSTNet, Transformer:**

```python
# 1. Tạo biến mà mô hình học (chỉ dùng để train, KHÔNG đưa vào feature X)
y_train_delta = train_fit['nextPrice'] - train_fit['Price']

# 2. Train mô hình trên delta
model.fit(X_train, y_train_delta)

# 3. Dự đoán delta (mạng nơ-ron: inverse_transform trước nếu đã scale delta)
delta_pred = model.predict(X_test)

# 4. Cộng lại thành giá, rồi mới tính metric
price_pred = test['Price'].values + delta_pred
results = evaluate(test['nextPrice'].values, price_pred, test['Price'].values)
```

Lưu ý khi áp dụng:

- `delta` **không** được đưa vào ma trận feature `X`, và `nextPrice` cũng không.
- Với mạng nơ-ron: nếu scale `delta` thì `fit` scaler **chỉ trên Train**, và nhớ `inverse_transform` trước khi cộng với `Price`.
- Mọi bảng kết quả ghi rõ `Target: delta` ở cột *Ghi chú* (riêng ARIMA ghi `d=1`).

**Cần chuẩn bị tâm lý:** sau khi xử lý đúng, nhiều mô hình có thể **chỉ ngang hoặc thậm chí kém hơn Naive một chút** (như gradient boosting ở mục 4.3: 2,86 so với 2,62). Điều này là bình thường với giá cổ phiếu theo ngày, vì giá hôm nay đã chứa gần hết thông tin cho ngày mai. Khi đó:

- Vẫn báo cáo trung thực, kèm Naive làm mốc.
- Dùng thêm **Directional Accuracy** (dự đoán đúng hướng tăng/giảm) để đánh giá. Với cách này, chỉ số đó có ý nghĩa rõ ràng hơn so với việc nhìn riêng MAE/RMSE.
- Phân tích nguyên nhân (giá có tính ngẫu nhiên cao, thông tin từ feature hạn chế, ...) cũng là một phần có giá trị của đồ án.

---

## 5. Weighted Average Ensemble

Kết hợp dự đoán của **5 mô hình cơ sở** (ARIMA, Ridge, LightGBM, LSTNet, Transformer) bằng trọng số.

**Công thức**

```
price_pred_ensemble = w_1 * price_pred_ARIMA  + w_2 * price_pred_Ridge
                    + w_3 * price_pred_LightGBM + w_4 * price_pred_LSTNet
                    + w_5 * price_pred_Transformer
với  w_i >= 0  và  w_1 + ... + w_5 = 1
```

Vì `price_pred_i = Price + delta_pred_i` và tổng trọng số bằng 1, kết hợp trên giá hay trên `delta` cho cùng một kết quả.

**Cách tính trọng số (đề xuất):** trọng số tỉ lệ nghịch với RMSE trên tập validation `train_val`:

```
w_i = (1 / RMSE_i) / sum_j (1 / RMSE_j)
```

**Quy ước để làm Ensemble được:**

- Mọi mô hình **huấn luyện trên `train_fit`** (90% đầu của Train) và dự đoán cho **`train_val`** (10% cuối của Train) và **Test**.
- Trọng số **chỉ tính từ `train_val`**. Tuyệt đối không dùng Test để chọn trọng số.
- Mỗi người lưu dự đoán của mô hình mình ra 2 file CSV: `pred_<tên_mô_hình>_val.csv` và `pred_<tên_mô_hình>_test.csv`, gồm cột `Date` và `price_pred` (giá `nextPrice` dự đoán, đã cộng lại).
- Các file phải **phủ đúng cùng các ngày** và cùng thứ tự thời gian. Mô hình dùng cửa sổ (LSTNet, Transformer) ghép đuôi của tập trước để không mất mẫu đầu, như quy ước ở `README.md`.
- Mô hình có seed ngẫu nhiên (LSTNet, Transformer) dùng **trung bình dự đoán qua 3 seed** để đưa vào Ensemble.

**Đánh giá Ensemble** (theo slide: độ chính xác và tính ổn định):

- Độ chính xác: các metric chung (MAE, RMSE, MAPE, R², DirAcc) trên Test.
- Tính ổn định: độ lệch chuẩn qua các seed (với mạng nơ-ron) và sai số theo từng giai đoạn của Test (ví dụ chia Test thành 4 đoạn bằng nhau), so với từng mô hình đơn lẻ.

---

## 6. Mẫu bảng kết quả cho các mô hình trong nhóm

Điền theo định dạng ở mục 8 của `README.md`, kết quả trên **tập Test** (đơn vị USD, tính sau khi cộng lại thành giá):

| Mô hình | Người phụ trách | Feature | Window | MAE | RMSE | MAPE (%) | R² | DirAcc (%) | Seed | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|
| Naive | – | – | – | 2,62 | 3,84 | 1,14 | | | – | Baseline (delta = 0) |
| ARIMA | A | Price | – | | | | | | – | `(p,1,q)` = ... |
| Ridge | B | Cơ sở | – | | | | | | 42 | Target: delta |
| LightGBM | C | Cơ sở | – | | | | | | 42 | Target: delta |
| LSTNet | D | Cơ sở | 30 | | | | | | 42 | Target: delta |
| Transformer | E | Cơ sở | 30 | | | | | | 42 | Target: delta |
| Weighted Average Ensemble | Chưa phân công | – | – | | | | | | – | Trọng số từ `train_val` |

---

## 7. Các điểm cần cả nhóm chốt

- [x] ~~Cách xử lý vấn đề ở mục 4~~ **Đã chốt:** học `nextPrice - Price`, cộng lại thành giá
- [ ] Ai phụ trách Weighted Average Ensemble
- [ ] Cách tính trọng số Ensemble (đề xuất ở mục 5: tỉ lệ nghịch với RMSE trên `train_val`)
- [ ] Có thêm chỉ báo kỹ thuật (RSI, MACD, MA, lợi suất trễ) vào feature hay không (xem `README.md`, mục 2.4)
- [ ] Dùng TensorFlow hay PyTorch cho LSTNet và Transformer
