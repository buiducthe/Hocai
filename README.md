# ⚖️ Hệ Thống Dự Báo Gian Lận Báo Cáo Tài Chính (Financial Statement Fraud Detection)

Ứng dụng Web tương tác được xây dựng trên nền tảng **Streamlit**, ứng dụng thuật toán Học máy **Logistic Regression** kết hợp cùng mô hình **8 biến Beneish M-Score** để phân tích, phát hiện sớm và cảnh báo rủi ro thao túng Báo cáo tài chính (BCTC) của doanh nghiệp.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📌 Mục lục
1. [Giới thiệu Dự án](#-giới-thiệu-dự-án)
2. [Các Tính Năng Nổi Bật](#-các-tính-năng-nổi-bật)
3. [8 Chỉ Số Tài Chính Beneish M-Score](#-8-chỉ-số-tài-chính-beneish-m-score)
4. [Cấu Trúc Thư Mục](#-cấu-trúc-thư-mục)
5. [Hướng Dẫn Cài Đặt & Chạy Cục Bộ (Local)](#-hướng-dẫn-cài-đặt--chạy-cục-bộ-local)
6. [Hướng Dẫn Đẩy Lên GitHub & Deploy Streamlit Cloud](#-hướng-dẫn-đẩy-lên-github--deploy-streamlit-cloud)
7. [Định Dạng File Dữ Liệu](#-định-dạng-file-dữ-liệu)

---

## 🎯 Giới thiệu Dự án

Thao túng lợi nhuận và gian lận BCTC luôn là mối quan tâm hàng đầu của kiểm toán viên, chuyên viên phân tích tín dụng và nhà đầu tư chứng khoán. Mô hình truyền thống Beneish M-Score (1999) sử dụng các trọng số cố định từ thị trường Mỹ, trong khi hệ thống này kết hợp:
- **Chuẩn hóa dữ liệu (`StandardScaler`)** giúp loại bỏ độ lệch đơn vị giữa các tỷ số.
- **Hồi quy Logistic (`LogisticRegression`)** được huấn luyện và hiệu chỉnh trực tiếp trên tập dữ liệu thực tế.
- **Tùy biến ngưỡng quyết định (Threshold Tuning)**: Trong kiểm toán, chi phí bỏ sót gian lận (**False Negative**) thường lớn hơn nhiều so với cảnh báo nhầm (**False Positive**). Ứng dụng cho phép người dùng chủ động điều chỉnh ngưỡng xác suất để tối ưu hóa độ nhạy (Recall).

---

## 🌟 Các Tính Năng Nổi Bật

Ứng dụng được chia thành 5 phân hệ chuyên sâu:

1. **📊 Tổng quan Dữ liệu & EDA (Exploratory Data Analysis)**:
   - Thống kê tổng số mẫu, tỷ lệ mẫu gian lận (Nhãn 1) và bình thường (Nhãn 0).
   - Biểu đồ tròn phân phối nhãn gian lận.
   - Ma trận tương quan (Correlation Heatmap) giữa 8 biến Beneish và biến mục tiêu `FRAUD_FLAG`.
   - Biểu đồ Boxplot và Histogram so sánh từng chỉ số giữa 2 nhóm.
2. **📈 Huấn luyện & Đánh giá Mô hình (Model Performance)**:
   - Bảng hệ số hồi quy ($\beta$), Odds Ratio ($e^\beta$) và phương trình toán học Logit đầy đủ.
   - Ma trận nhầm lẫn tương tác (Interactive Confusion Matrix) với số lượng cụ thể của TN, FP, FN, TP.
   - Thước đo hiệu năng toàn diện: **Accuracy**, **Precision**, **Recall**, **F1-Score**, **Specificity**, **FPR**, **FNR**.
   - Đường cong ROC và giá trị ROC-AUC.
   - Thanh trượt điều chỉnh ngưỡng xác suất phân loại (Threshold Slider) cập nhật kết quả tức thì.
   - Xuất toàn bộ kết quả ra file **Excel (.xlsx)** gồm 5 sheet chuyên nghiệp.
3. **🔍 Dự báo Doanh nghiệp Đơn lẻ (Single Prediction)**:
   - Giao diện nhập 8 chỉ số của một doanh nghiệp cụ thể kèm diễn giải nghiệp vụ.
   - Nút nạp nhanh cấu hình mẫu: Doanh nghiệp an toàn, Doanh nghiệp rủi ro cao, Trung bình mẫu.
   - Đồng hồ đo xác suất rủi ro (Gauge Chart).
   - Tính toán đối chiếu song song với mô hình Beneish M-Score nguyên bản ($M > -2.22$).
   - Biểu đồ cột so sánh chỉ số của doanh nghiệp với mức trung bình của tập dữ liệu.
4. **📁 Dự báo Hàng loạt (Batch Prediction)**:
   - Hỗ trợ tải file mẫu template CSV.
   - Tải lên danh sách nhiều công ty cùng lúc (`.csv` hoặc `.xlsx`).
   - Tự động quét, phân loại rủi ro và gắn nhãn cảnh báo.
   - Xuất kết quả dự báo ra file CSV hoặc Excel.
5. **📖 Hướng dẫn & Cơ sở Lý thuyết**:
   - Cung cấp kiến thức nghiệp vụ kiểm toán và tài chính về 8 chỉ số Beneish.

---

## 📐 8 Chỉ Số Tài Chính Beneish M-Score

| Ký hiệu | Tên chỉ số | Ý nghĩa kinh tế / Dấu hiệu rủi ro |
| :--- | :--- | :--- |
| **DSRI** | Days Sales in Receivables Index | Số ngày thu tiền khách hàng. DSRI > 1 cho thấy nợ phải thu tăng nhanh hơn doanh thu, nguy cơ ghi nhận doanh thu ảo. |
| **GMI** | Gross Margin Index | Biên lợi nhuận gộp. GMI > 1 phản ánh biên lợi nhuận sụt giảm, tạo động cơ thổi phồng lợi nhuận. |
| **AQI** | Asset Quality Index | Chất lượng tài sản. AQI > 1 biểu thị xu hướng vốn hóa chi phí vào tài sản kém chất lượng. |
| **SGI** | Sales Growth Index | Tốc độ tăng trưởng doanh thu. Tăng trưởng quá nóng tạo áp lực duy trì kỳ vọng thị trường. |
| **DEPI** | Depreciation Index | Tỷ lệ khấu hao. DEPI > 1 cho thấy công ty giảm tốc độ trích khấu hao để làm đẹp lợi nhuận ngắn hạn. |
| **SGAI** | Sales, General & Admin Expense Index | Chi phí bán hàng & QLDN. SGAI > 1 cho thấy hiệu quả kiểm soát chi phí hoạt động suy giảm. |
| **TATA** | Total Accruals to Total Assets | Tổng biến dồn tích trên tổng tài sản. TATA cao phản ánh lợi nhuận không đi kèm dòng tiền thực. |
| **LVGI** | Leverage Index | Đòn bẩy tài chính. LVGI > 1 cho thấy tỷ lệ nợ vay gia tăng, tăng rủi ro thanh khoản. |

---

## 📁 Cấu Trúc Thư Mục

```text
├── app.py                             # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt                   # Danh sách thư viện Python cần thiết
├── README.md                          # Tài liệu hướng dẫn sử dụng và triển khai
├── MScore_data.csv                    # Tập dữ liệu mẫu 8 chỉ số Beneish & nhãn FRAUD_FLAG
└── logistic_regression_mscore_colab.py # Mã nguồn Colab gốc dùng để đối chiếu
```

---

## 💻 Hướng Dẫn Cài Đặt & Chạy Cục Bộ (Local)

### 1. Yêu cầu hệ thống
- Máy tính đã cài sẵn **Python 3.9** trở lên.
- Trình duyệt web hiện đại (Chrome, Edge, Firefox, Brave,...).

### 2. Các bước thực hiện

1. **Mở Terminal / PowerShell** tại thư mục dự án:
   ```bash
   cd "c:\Users\Admin\Desktop\hoc AI"
   ```

2. **(Khuyến nghị) Tạo và kích hoạt môi trường ảo:**
   - Trên Windows:
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - Trên macOS / Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Khởi chạy ứng dụng Streamlit:**
   ```bash
   streamlit run app.py
   ```

5. **Truy cập ứng dụng:**
   Trình duyệt sẽ tự động mở địa chỉ: `http://localhost:8501`.

---

## 🚀 Hướng Dẫn Đẩy Lên GitHub & Deploy Streamlit Cloud

### Bước 1: Tạo Repository trên GitHub
1. Truy cập [GitHub](https://github.com) và đăng nhập tài khoản.
2. Nhấn nút **New** (Tạo repo mới).
3. Đặt tên Repository, ví dụ: `fraud-detection-beneish-mscore`.
4. Chọn chế độ **Public**, không cần tích chọn README (vì bạn đã có sẵn file README.md này).
5. Nhấn **Create repository**.

### Bước 2: Đẩy Code lên GitHub từ máy tính
Mở Terminal / PowerShell tại thư mục dự án và thực hiện tuần tự các lệnh sau:

```bash
git init
git add app.py requirements.txt README.md MScore_data.csv
git commit -m "Khoi tao web app du bao gian lan BCTC tren Streamlit"
git branch -M main
git remote add origin https://github.com/<TEN-TAI-KHOAN-GITHUB>/fraud-detection-beneish-mscore.git
git push -u origin main
```
*(Thay thế `<TEN-TAI-KHOAN-GITHUB>` bằng username GitHub của bạn).*

> **Lưu ý quan trọng:** Đảm bảo file dữ liệu `MScore_data.csv` được đẩy lên GitHub cùng với `app.py` và `requirements.txt`. Khi ứng dụng khởi động trên Streamlit Cloud, nó sẽ tự động đọc file này làm dữ liệu mẫu huấn luyện mặc định!

### Bước 3: Deploy lên Streamlit Community Cloud (Hoàn toàn miễn phí)
1. Truy cập [share.streamlit.io](https://share.streamlit.io) và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấn nút **New app**.
3. Điền các thông tin:
   - **Repository:** Chọn repository vừa tạo (ví dụ: `your-username/fraud-detection-beneish-mscore`).
   - **Branch:** Chọn `main`.
   - **Main file path:** Nhập `app.py`.
4. Nhấn **Deploy!**
5. Chờ 1 - 2 phút để Streamlit Cloud tự động cài đặt các thư viện trong `requirements.txt` và khởi chạy web app.
6. Sau khi hoàn tất, bạn sẽ nhận được một đường link công khai (URL) dạng: `https://<ten-ung-dung>.streamlit.app` để chia sẻ cho mọi người cùng sử dụng!

---

## 📊 Định Dạng File Dữ Liệu

### 1. Dữ liệu huấn luyện (`MScore_data.csv`)
Phải bao gồm 9 cột chính xác:
- 8 cột chỉ số: `DSRI`, `GMI`, `AQI`, `SGI`, `DEPI`, `SGAI`, `TATA`, `LVGI`
- 1 cột nhãn: `FRAUD_FLAG` (chỉ nhận giá trị `0` là không gian lận hoặc `1` là có gian lận).

### 2. Dữ liệu dự báo hàng loạt (Batch Prediction)
Chỉ cần có 8 cột chỉ số Beneish nêu trên. Có thể có thêm cột tên công ty như `COMPANY_NAME` hoặc mã cổ phiếu `TICKER`. Hệ thống sẽ giữ nguyên các cột định danh này và thêm các cột kết quả dự báo xác suất và kết luận.

---

## 📜 Giấy Phép & Bản Quyền
Dự án được phân phối dưới giấy phép **MIT License**. Mọi người được tự do sử dụng, chỉnh sửa và ứng dụng vào mục đích học tập hoặc nghiên cứu học thuật.
