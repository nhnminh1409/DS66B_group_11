# Dự đoán khách hàng rời mạng Telco

Nghiên cứu thực nghiệm về dự đoán **Customer Churn** trên bộ dữ liệu Telco Customer Churn. Đề tài so sánh các pipeline phân loại nhị phân, xử lý mất cân bằng lớp, lựa chọn ngưỡng dự đoán và diễn giải mô hình.

> Đây là bài tập môn Statistical Learning của nhóm DS66B - Nhóm 11. Dự án tập trung vào việc áp dụng và đánh giá các phương pháp thống kê/học máy, không đề xuất một thuật toán mới và không khẳng định quan hệ nhân quả.

## Repository structure

```text
.
├── master_churn_prediction.ipynb   # Notebook phân tích chính
├── requirements.txt                # Các thư viện Python cần thiết
├── data/
│   ├── raw/                        # Dataset local, không commit
│   └── processed/                  # Dữ liệu xử lý sinh ra từ notebook
├── artifacts/                      # Model/bảng kết quả sinh ra từ notebook
└── reports/
    └── figures/                    # Hình sinh ra từ notebook
```

Notebook cũng có thể đọc file CSV đặt ngay tại thư mục gốc của repository.

## Dataset

Dataset không được lưu trong repository. Tải dữ liệu từ nguồn Kaggle được sử dụng trong bài:

- [Customer Churn Prediction - Kaggle](https://www.kaggle.com/code/bhartiprasad17/customer-churn-prediction)

Sau khi tải, đặt file CSV vào một trong các vị trí sau:

```text
data/raw/telco_customer_churn.csv
# hoặc
./telco_customer_churn.csv
# hoặc
./WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Notebook tìm lần lượt các đường dẫn trên. File dữ liệu cần có các cột chính như `customerID`, `tenure`, `MonthlyCharges`, `TotalCharges` và `Churn`.

## Cài đặt

Yêu cầu Python 3.10+ được khuyến nghị.

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Cài đặt thư viện:

```bash
python -m pip install -r requirements.txt
```

## Chạy notebook

Mở Jupyter Notebook:

```bash
jupyter notebook
```

Sau đó mở:

```text
master_churn_prediction.ipynb
```

Chạy từ đầu bằng:

```text
Kernel → Restart & Run All
```

Notebook sử dụng `random_state = 42`, tỷ lệ Test 20% và Stratified 5-Fold Cross-Validation để kết quả có thể tái lập trong cùng môi trường thư viện.

## Quy trình phân tích

1. Kiểm tra dữ liệu thô và tính toàn vẹn của các cột.
2. Làm sạch `TotalCharges` và xử lý các nhãn dịch vụ dư thừa.
3. Chia dữ liệu Train/Test theo phương pháp stratified split.
4. Phân tích thống kê mô tả, kiểm định Chi-square và Cramér's V.
5. Tạo một số đặc trưng nghiệp vụ như `Charge_Ratio` và `Tenure_Cohort`.
6. Xây dựng pipeline preprocessing và xử lý mất cân bằng lớp.
7. So sánh Dummy Classifier, Logistic Regression, KNN, Decision Tree, Random Forest và XGBoost.
8. Tinh chỉnh siêu tham số bằng cross-validation trên tập Train.
9. Điều chỉnh threshold và đánh giá trên tập Test chỉ sau khi chọn mô hình.
10. Phân tích cost scenario, Odds Ratio và SHAP để hỗ trợ diễn giải.

## Kết quả chính

Mô hình được chọn theo điểm **F1 trên các fold validation của tập Train**, không chọn theo kết quả Test.

Trong notebook, cấu hình `RF-tuned` được chọn làm Champion:

| Metric | Cross-validation result |
|---|---:|
| F1 mean | 0.6393 |
| F1 standard deviation | 0.0171 |
| ROC-AUC mean | 0.8433 |

Khoảng cách giữa các mô hình đứng đầu tương đối nhỏ. Vì vậy, Random Forest được giữ làm Champion theo tiêu chí xếp hạng đã định trước, không nên diễn giải là vượt trội tuyệt đối trong mọi khía cạnh.

## Diễn giải và giới hạn

- Accuracy không phải chỉ số chính vì tỷ lệ churn thấp hơn lớp không churn; cần xem đồng thời Precision, Recall, F1, ROC-AUC và PR-AUC.
- Mô hình Champion được lựa chọn bằng cross-validation trên Train; Test được giữ cho đánh giá cuối cùng.
- Điểm CV sau tuning có thể hơi lạc quan vì quá trình chọn siêu tham số và đánh giá dùng chung các fold. Nghiên cứu không sử dụng nested cross-validation.
- Cost matrix chỉ là một kịch bản chi phí giả định, không phải ước lượng ROI thực tế.
- Odds Ratio và SHAP mô tả association/conditional association và mức độ đóng góp của đặc trưng trong mô hình; chúng không chứng minh quan hệ nhân quả.
- VIF cao ở một số biến phản ánh đa cộng tuyến trong dữ liệu. Vì vậy, Odds Ratio cần được diễn giải thận trọng và không được xem là tác động độc lập hoàn toàn đáng tin cậy.
- SMOTE chỉ là một cấu hình/ablation để so sánh; không mặc định rằng dữ liệu tổng hợp luôn tốt hơn class weighting.

## Thành viên

- DS66B - Nhóm 11

## License

Dự án phục vụ mục đích học tập. Dataset được cung cấp bởi nguồn bên ngoài và không được phân phối lại trong repository này.

## Tài liệu tham khảo

- [Repository](https://github.com/nhnminh1409/DS66B_group_11)
- [Kaggle notebook/source link](https://www.kaggle.com/code/bhartiprasad17/customer-churn-prediction)
- Telco Customer Churn dataset, thường được phân phối với tên `WA_Fn-UseC_-Telco-Customer-Churn.csv`

---

**Lưu ý:** Không commit dataset, model đã lưu, hình ảnh sinh ra hoặc các file trung gian vào repository. Các loại file này đã được thêm vào `.gitignore`.
