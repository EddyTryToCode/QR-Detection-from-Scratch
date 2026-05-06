# Bổ sung Định dạng Đầu vào / Đầu ra và Tiêu chí Chấm điểm Tự động

> Tài liệu này bổ sung phần còn thiếu trong đề bài gốc: quy cách chi tiết của
> file đầu vào, file đầu ra, và quy trình chấm điểm tự động.

---

## 1. Cấu trúc thư mục nộp bài

```
submission/
├── main.py              # file chính, bắt buộc
├── requirements.txt     # danh sách thư viện Python cần cài
└── output.csv           # file kết quả (sinh ra khi chạy main.py)
```

> Báo cáo PDF nộp riêng theo quy định của lớp.

---

## 2. Định dạng file đầu vào (public.csv / private.csv)

### 2.1 Cấu trúc

File CSV chuẩn (UTF-8, dấu phẩy `,` làm separator, có header).

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `image_id` | string | Định danh duy nhất của ảnh (tên file không có phần mở rộng) |
| `image_path` | string | Đường dẫn tương đối tới file ảnh (tính từ vị trí file CSV) |

### 2.2 Ví dụ

```csv
image_id,image_path
2656508531_jpg.rf.e3471da5,valid/2656508531_jpg.rf.e3471da5.jpg
2879826877_jpg.rf.e5a0a919,valid/2879826877_jpg.rf.e5a0a919.jpg
IMG_20220601_162857_jpg.rf,valid/IMG_20220601_162857_jpg.rf.e37733db.jpg
```

### 2.3 Lưu ý

- Mỗi hàng (ngoài header) tương ứng đúng một ảnh cần xử lý.
- `image_path` có thể là ảnh `.jpg`, `.jpeg`, hoặc `.png`.
- Chương trình **không được** giả định tên file hay đường dẫn cụ thể; phải
  đọc từ file CSV được truyền vào qua `--data`.

---

## 3. Định dạng file đầu ra (output.csv)

### 3.1 Cấu trúc

Chương trình phải ghi kết quả ra file `output.csv` đặt **cùng thư mục với
`main.py`**. File có định dạng CSV chuẩn (UTF-8, dấu phẩy, có header), **một
hàng cho mỗi QR code được phát hiện**.

| Cột | Kiểu | Bắt buộc | Mô tả |
|-----|------|----------|-------|
| `image_id` | string | Có | Phải khớp với `image_id` trong file đầu vào |
| `qr_index` | int hoặc rỗng | Có | Thứ tự của QR trong ảnh (0-based). Để rỗng hoặc `-1` nếu ảnh không có QR |
| `x0` | float | Có* | Tọa độ x góc trên-trái (top-left) |
| `y0` | float | Có* | Tọa độ y góc trên-trái |
| `x1` | float | Có* | Tọa độ x góc trên-phải (top-right) |
| `y1` | float | Có* | Tọa độ y góc trên-phải |
| `x2` | float | Có* | Tọa độ x góc dưới-phải (bottom-right) |
| `y2` | float | Có* | Tọa độ y góc dưới-phải |
| `x3` | float | Có* | Tọa độ x góc dưới-trái (bottom-left) |
| `y3` | float | Có* | Tọa độ y góc dưới-trái |
| `content` | string | Không | Nội dung giải mã của QR (để rỗng nếu không giải mã) |

> \* Các trường tọa độ để **rỗng** khi `qr_index` là rỗng hoặc `-1`.

### 3.2 Quy ước tọa độ 4 góc

```
Hệ tọa độ ảnh:  gốc (0,0) ở góc trên-trái
                 x tăng sang phải
                 y tăng xuống dưới

  (x0,y0) -------- (x1,y1)
     |                  |
     |      QR code     |
     |                  |
  (x3,y3) -------- (x2,y2)
```

Đối với QR code **không bị xoay** (trường hợp phổ biến nhất):
- `(x0, y0)` = góc trên-trái
- `(x1, y1)` = góc trên-phải
- `(x2, y2)` = góc dưới-phải
- `(x3, y3)` = góc dưới-trái

Đối với QR code **bị xoay hoặc biến dạng phối cảnh**: 4 điểm vẫn phải tạo
thành tứ giác lồi bao quanh QR, theo thứ tự kim đồng hồ bắt đầu từ góc gần
với góc trên-trái nhất.

> **Lưu ý về bounding box hướng trục:** Nếu chương trình chỉ phát hiện được
> bounding box hướng trục (axis-aligned), tức `[x, y, w, h]`, thì chuyển sang
> 4 góc như sau:
> ```
> x0, y0 = x,     y
> x1, y1 = x + w, y
> x2, y2 = x + w, y + h
> x3, y3 = x,     y + h
> ```

### 3.3 Xử lý ảnh không có QR code

Với mỗi ảnh **không phát hiện được** QR code nào, chương trình **vẫn phải**
ghi một hàng vào `output.csv` để xác nhận đã xử lý ảnh đó:

```csv
image_id,qr_index,x0,y0,x1,y1,x2,y2,x3,y3,content
img_without_qr,,,,,,,,,,
```
(tất cả các trường từ `qr_index` trở đi đều để rỗng)

### 3.4 Ví dụ đầy đủ

```csv
image_id,qr_index,x0,y0,x1,y1,x2,y2,x3,y3,content
2656508531_jpg.rf.e3471da5,0,75,82,161,82,161,148,75,148,https://example.com
2879826877_jpg.rf.e5a0a919,0,103,309,183,309,183,412,103,412,
2879826877_jpg.rf.e5a0a919,1,315,419,403,419,403,529,315,529,Hello World
IMG_20220601_162857_jpg.rf,,,,,,,,,,
lotsimage007_jpg.rf.8c19a5,0,290,173,319,173,319,195,290,195,
lotsimage007_jpg.rf.8c19a5,1,340,210,380,210,380,250,340,250,QR Content 2
```

Giải thích:
- Ảnh `2656508531` có 1 QR, đã giải mã được nội dung.
- Ảnh `2879826877` có 2 QR; QR thứ nhất chưa giải mã (để rỗng).
- Ảnh `IMG_20220601_162857` không có QR nào.
- Ảnh `lotsimage007` có nhiều QR (chỉ 2 hàng ví dụ).

### 3.5 Yêu cầu kỹ thuật

- **Header bắt buộc** và phải đúng tên cột như trên (phân biệt hoa thường).
- Không được có khoảng trắng thừa quanh giá trị.
- Nếu `content` chứa dấu phẩy, dấu nháy kép, hay xuống dòng thì phải được bao
  trong dấu nháy kép theo chuẩn CSV (RFC 4180).
- Tọa độ là số thực (float); giá trị nguyên cũng được chấp nhận.
- Thứ tự các hàng không quan trọng đối với grader.

---

## 4. Cách chạy chương trình

```bash
# Cài thư viện (lần đầu)
pip install -r requirements.txt

# Chạy trên tập public
python main.py --data public.csv

# Chạy trên tập private (giảng viên thực hiện)
python main.py --data private.csv
```

Sau khi chạy xong, file `output.csv` sẽ được tạo tại thư mục chứa `main.py`.

---

## 5. Quy trình chấm điểm tự động

### 5.1 Thuật toán matching (Greedy IoU Matching)

Để so sánh vị trí dự đoán với ground truth, grader dùng **Greedy Matching**
theo giá trị IoU (Intersection over Union) giữa các tứ giác:

```
Với mỗi ảnh:
  Cho mỗi bounding box dự đoán P_i (theo thứ tự):
    Tìm GT box G_j chưa được ghép có IoU(P_i, G_j) cao nhất
    Nếu IoU cao nhất ≥ IoU_threshold → ghép (TP)
    Ngược lại                        → FP

  Các GT box chưa được ghép → FN
```

**Ngưỡng IoU mặc định**: `0.5`

### 5.2 Tính IoU cho tứ giác

IoU được tính trên diện tích thực của tứ giác (không phải bounding rectangle),
dùng thuật toán **Sutherland–Hodgman** để tính giao giữa hai đa giác lồi.

```
             Diện tích giao (Intersection)
IoU = ─────────────────────────────────────────
        Diện tích P + Diện tích G − Diện tích giao
```

### 5.3 Metric đánh giá

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| Precision | TP / (TP + FP) | Tỷ lệ dự đoán đúng trong tổng số dự đoán |
| Recall | TP / (TP + FN) | Tỷ lệ QR code thực sự được tìm thấy |
| **F1 Score** | 2·P·R / (P + R) | **Chỉ số chính để xếp hạng** |

Trong đó:
- **TP** (True Positive): dự đoán khớp với GT (IoU ≥ 0.5)
- **FP** (False Positive): dự đoán không khớp GT nào
- **FN** (False Negative): GT không có dự đoán nào khớp

### 5.4 Đánh giá nội dung QR (tùy chọn cộng điểm)

Với mỗi cặp (TP), nếu cột `content` trong output không rỗng:

```
Content_Accuracy = số TP có content khớp chính xác / tổng số TP
```

"Khớp chính xác" là so sánh chuỗi sau khi đã strip() khoảng trắng hai đầu và
chuẩn hóa encoding UTF-8. Không phân biệt hoa thường (case-insensitive).

### 5.5 Đánh giá tốc độ

- Tốc độ được đo bằng **tổng thời gian chạy** (wall-clock time) của lệnh
  `python main.py --data private.csv` trên máy chấm của giảng viên.
- Đơn vị: giây; chuẩn hóa trên số lượng ảnh → **giây/ảnh**.
- Máy chấm sử dụng CPU (không có GPU).

### 5.6 Điểm tổng hợp

```
Score = 0,7 × F1  +  0,3 × Speed_Score
```

Trong đó `Speed_Score` được tính theo thứ hạng (rank-based normalization) so
với các nhóm khác.

---
