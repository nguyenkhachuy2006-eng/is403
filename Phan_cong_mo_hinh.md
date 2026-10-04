# Danh sách mô hình & phân công (tham khảo)

File này bổ sung cho `README.md`. Mọi quy ước về **dữ liệu, chia tập, tiền xử lý, metric, seed** vẫn theo `README.md`. File này nói về **mô hình nào sẽ được chạy**, **gợi ý ai làm gì** và **một vấn đề quan trọng về dữ liệu cần cả nhóm thống nhất cách xử lý** (mục 4).

> ✅ **ĐÃ CHỐT:** mọi mô hình học **`delta = nextPrice - Price`** (mức thay đổi giá từ hôm nay sang ngày mai), sau đó cộng lại `nextPrice_dự_đoán = Price + delta_dự_đoán` để tính metric trên giá USD. Chi tiết ở mục 4.

> ⚠️ **Phân công dưới đây chỉ mang tính tham khảo.** Với mạng nơ-ron, mỗi người hoàn toàn có thể kết hợp nhiều kiến trúc (ví dụ CNN + LSTM, LSTM + Attention, GRU + Transformer encoder, ...) và không bắt buộc phải theo đúng danh sách. Điều kiện duy nhất: vẫn tuân thủ quy ước chung trong `README.md` và ghi rõ kiến trúc đã dùng ở cột *Ghi chú* của bảng kết quả.

---

## 1. Các mô hình sẽ chạy

| Nhóm | Mô hình | Dạng đầu vào |
|---|---|---|
| Mạng nơ-ron | **Transformer** | Cửa sổ 3D `(mẫu, WINDOW, feature)` |
| Mạng nơ-ron | **LSTNet** | Cửa sổ 3D `(mẫu, WINDOW, feature)` |
| Mạng nơ-ron | **LSTM** | Cửa sổ 3D `(mẫu, WINDOW, feature)` |
| Mạng nơ-ron | **GRU** | Cửa sổ 3D `(mẫu, WINDOW, feature)` |
| Mạng nơ-ron | **SimpleRNN** | Cửa sổ 3D `(mẫu, WINDOW, feature)` |
| Machine Learning | **Random Forest** | Bảng 2D `(mẫu, feature)` |
| Machine Learning | **XGBoost** | Bảng 2D `(mẫu, feature)` |
| Machine Learning | **LightGBM** | Bảng 2D `(mẫu, feature)` |
| Baseline | Naive (`nextPrice = Price`) | – |

---

## 2. Phân công gợi ý (5 thành viên)

Transformer và LSTNet mỗi mô hình do **một người** đảm nhận. Ba người còn lại, mỗi người phụ trách **1 mạng nơ-ron + 1 mô hình ML**.

| Thành viên | Mạng nơ-ron | Machine Learning |
|---|---|---|
| A | LSTM | Random Forest |
| B | GRU | XGBoost |
| C | SimpleRNN | LightGBM |
| D | Transformer | – |
| E | LSTNet | – |

- Cách ghép cặp NN – ML ở trên chỉ là ví dụ, nhóm có thể hoán đổi tuỳ ý.
- Naive baseline dùng **chung một hàm** trong code chung nên không cần giao riêng. Linear Regression (nếu muốn thêm làm baseline) chưa phân công.

---

## 3. Gợi ý cho từng mô hình

### Mô hình ML (đầu vào dạng bảng)

| Mô hình | Siêu tham số nên thử | Ghi chú |
|---|---|---|
| Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` | Không cần chuẩn hoá, nhưng giữ chung quy trình với nhóm cho đồng nhất |
| XGBoost | `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree` | Có thể dùng `early_stopping_rounds` trên phần validation cuối Train |
| LightGBM | `num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`, `feature_fraction` | Tương tự XGBoost |

- Chỉnh siêu tham số bằng **`TimeSeriesSplit`** trên tập Train (theo `README.md`).
- Mô hình ML không tự nhìn thấy chuỗi thời gian. Nếu muốn, có thể thêm **lag feature** (ví dụ `Price_lag1`, `Price_lag2`, ..., hoặc trung bình động) và ghi rõ trong bảng kết quả.

### Mô hình mạng nơ-ron (đầu vào dạng cửa sổ)

| Mô hình | Đặc điểm | Ghi chú |
|---|---|---|
| SimpleRNN | Mạng hồi quy cơ bản, dễ bị vanishing gradient với cửa sổ dài | Dùng làm mốc so sánh với LSTM / GRU |
| LSTM | Có cổng nhớ, học phụ thuộc dài hạn | Thử số lớp, số units, dropout |
| GRU | Gọn hơn LSTM, ít tham số hơn | Thử số lớp, số units, dropout |
| LSTNet | Kết hợp Conv1D, GRU, skip-GRU và thành phần tự hồi quy (highway) | Thiết kế cho chuỗi đa biến, thử kernel size, hidden size, skip |
| Transformer | Dùng self-attention, cần positional encoding | Thử số head, số lớp encoder, `d_model`, dropout |

- Cửa sổ mặc định `WINDOW = 30` như trong `README.md`.
- Dùng EarlyStopping theo `val_loss` như quy ước chung.

---

## 4. Vấn đề quan trọng: giá trong tập Test vượt khỏi phạm vi tập Train

Đây là vấn đề **ảnh hưởng đến tất cả mô hình**. Nhóm đã thống nhất cách xử lý ở mục 4.4. Các mục 4.1 đến 4.3 giải thích lý do.

### 4.1. Hiện tượng

Dữ liệu là chuỗi thời gian và giá Apple tăng mạnh theo thời gian, nên khi chia 80/20 theo thứ tự thời gian:

| | Khoảng giá `Price` (USD) |
|---|---|
| Tập Train (04/10/2010 → 14/07/2023) | 9,95 → **193,97** |
| Tập Test (17/07/2023 → 28/09/2026) | 165,00 → **341,07** |

Khoảng **74% số dòng trong tập Test** có `Price` cao hơn mức cao nhất mà mô hình từng thấy trong lúc train. Nói cách khác, đa số ngày Test nằm ở vùng giá mô hình **chưa bao giờ được học**.

### 4.2. Vì sao đây là vấn đề

**Với Random Forest, XGBoost, LightGBM (mô hình dựa trên cây quyết định):**
- Mỗi lá của cây trả về một giá trị lấy từ dữ liệu train. Vì vậy dự đoán **không thể vượt quá** giá trị lớn nhất của target trong tập Train (≈ 194 USD).
- Nếu ta bắt mô hình dự đoán thẳng `nextPrice`, thì khi giá thực tế lên 250, 300, 340 USD, mô hình vẫn chỉ trả về khoảng 190 USD. Sai số sẽ lớn dần theo thời gian, không phải vì mô hình "học dở" mà vì **về mặt cấu trúc nó không thể dự đoán vùng giá đó**.

**Với mạng nơ-ron (MLP, LSTM, GRU, SimpleRNN, LSTNet, Transformer):**
- Scaler được `fit` trên Train nên giá Test sau khi scale sẽ vượt ra ngoài khoảng đã quen (ví dụ với `MinMaxScaler`, giá trị > 1). Các hàm kích hoạt như `tanh`, `sigmoid` trong LSTM/GRU bị bão hoà, và mạng chưa từng thấy đầu vào ở vùng này nên dự đoán kém ổn định.
- Không hỏng cứng như mô hình cây, nhưng chất lượng thường giảm đáng kể.

**Với mô hình tuyến tính (Linear Regression, Ridge):**
- Ngoại suy được nên ít bị ảnh hưởng hơn (xem bảng ở mục 4.3).

### 4.3. Thực nghiệm nhanh để minh hoạ

Thử nghiệm nhanh với cấu hình đơn giản (Random Forest 200 cây, `random_state=42`, feature cơ sở theo `README.md`, chia 80/20 theo thời gian, chưa tinh chỉnh) để nhóm thấy mức độ chênh lệch. Metric tính trên giá USD của tập Test:

| Cách làm | MAE | RMSE | Nhận xét |
|---|---|---|---|
| Naive (`nextPrice = Price`) | 2,62 | 3,84 | Mốc tham chiếu |
| Random Forest, dự đoán thẳng `nextPrice` | **42,47** | **58,29** | Dự đoán lớn nhất chỉ ≈ 192,5, đúng như phân tích ở trên |
| Random Forest, học `nextPrice - Price` rồi cộng lại | 2,84 | 4,18 | Sai số giảm hơn 10 lần |
| Linear Regression, dự đoán thẳng `nextPrice` | 2,69 | 3,95 | Mô hình tuyến tính ngoại suy được |
| Linear Regression, học `nextPrice - Price` rồi cộng lại | 2,69 | 3,95 | Gần như không đổi |

Kết luận từ thực nghiệm:
- Mô hình cây dự đoán thẳng `nextPrice` cho kết quả **tệ hơn Naive hàng chục lần**, và nếu không biết nguyên nhân, nhóm sẽ dễ kết luận nhầm là "ML không phù hợp".
- Chỉ cần đổi **cách biểu diễn target** là sai số giảm từ 42,47 xuống 2,84.

### 4.4. Cách xử lý đã chốt: học `delta = nextPrice - Price`

**Cả nhóm dùng cách này cho TẤT CẢ mô hình** (cả ML và mạng nơ-ron) để kết quả so sánh được công bằng. Cụ thể:

| Nội dung | Quy ước đã chốt |
|---|---|
| Target chính thức của đồ án | Cột `nextPrice` (không đổi) |
| Biến mà mô hình học | **`delta = nextPrice - Price`** |
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

**Quy trình chung cho mọi mô hình:**

```python
# 1. Tạo biến mà mô hình học (chỉ dùng để train, KHÔNG đưa vào feature X)
y_train_delta = train['nextPrice'] - train['Price']

# 2. Train mô hình trên delta
model.fit(X_train, y_train_delta)

# 3. Dự đoán delta trên Test (mạng nơ-ron: inverse_transform trước nếu đã scale delta)
delta_pred = model.predict(X_test)

# 4. Cộng lại thành giá, rồi mới tính metric
price_pred = test['Price'].values + delta_pred
results = evaluate(test['nextPrice'].values, price_pred, test['Price'].values)
```

Lưu ý khi áp dụng:

- `delta` **không** được đưa vào ma trận feature `X`, và `nextPrice` cũng không.
- Với mạng nơ-ron: nếu scale `delta` thì `fit` scaler **chỉ trên Train**, và nhớ `inverse_transform` trước khi cộng với `Price`.
- Mọi bảng kết quả ghi rõ `Target: delta` ở cột *Ghi chú*.

**Cần chuẩn bị tâm lý:** sau khi xử lý đúng, nhiều mô hình có thể **chỉ ngang hoặc thậm chí kém hơn Naive một chút** (như Random Forest ở mục 4.3: 2,84 so với 2,62). Điều này là bình thường với giá cổ phiếu theo ngày, vì giá hôm nay đã chứa gần hết thông tin cho ngày mai. Khi đó:

- Vẫn báo cáo trung thực, kèm Naive làm mốc.
- Dùng thêm **Directional Accuracy** (dự đoán đúng hướng tăng/giảm) để đánh giá. Với cách này, chỉ số này có ý nghĩa rõ ràng hơn so với việc nhìn riêng MAE/RMSE.
- Phân tích nguyên nhân (giá có tính ngẫu nhiên cao, thông tin từ feature hạn chế, ...) cũng là một phần có giá trị của đồ án.

---

## 5. Mẫu bảng kết quả cho các mô hình trong nhóm

Điền theo định dạng ở mục 8 của `README.md`, kết quả trên **tập Test** (đơn vị USD, tính sau khi cộng lại thành giá):

| Mô hình | Người phụ trách | Feature | Window | MAE | RMSE | MAPE (%) | R² | DirAcc (%) | Seed | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|
| Naive | – | – | – | 2,62 | 3,84 | 1,14 | | | – | Baseline |
| Random Forest | A | Cơ sở | – | | | | | | 42 | Target: delta |
| XGBoost | B | Cơ sở | – | | | | | | 42 | Target: delta |
| LightGBM | C | Cơ sở | – | | | | | | 42 | Target: delta |
| LSTM | A | Cơ sở | 30 | | | | | | 42 | Target: delta |
| GRU | B | Cơ sở | 30 | | | | | | 42 | Target: delta |
| SimpleRNN | C | Cơ sở | 30 | | | | | | 42 | Target: delta |
| Transformer | D | Cơ sở | 30 | | | | | | 42 | Target: delta |
| LSTNet | E | Cơ sở | 30 | | | | | | 42 | Target: delta |

---

## 6. Các điểm cần cả nhóm chốt

- [x] ~~Cách xử lý vấn đề ở mục 4~~ **Đã chốt:** học `nextPrice - Price`, cộng lại thành giá
- [ ] Có thêm lag feature / trung bình động cho mô hình ML hay không
- [ ] Có thêm Linear Regression làm baseline hay không, và ai phụ trách
- [ ] Dùng TensorFlow hay PyTorch cho mạng nơ-ron
