import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)

def evaluate_model(y_true, y_pred, y_proba=None) -> dict:
    """
    Tính toán các metric phân loại cho mô hình dự đoán Churn.
    Lớp dương (Churn) được mặc định là nhãn 1.
    
    Args:
        y_true: Nhãn thực tế (0 hoặc 1).
        y_pred: Nhãn dự đoán (0 hoặc 1).
        y_proba: Xác suất dự đoán lớp 1 (tùy chọn, để tính roc_auc).
        
    Returns:
        dict: Chứa accuracy, precision, recall, f1, macro_f1, roc_auc.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average='macro', zero_division=0)
    }
    
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["roc_auc"] = None
        
    return metrics

def evaluate_pipeline(pipeline, X, y) -> dict:
    """
    Đánh giá một pipeline/mô hình trên tập dữ liệu X, y.
    Tự động gọi predict và predict_proba (nếu có).
    
    Args:
        pipeline: Mô hình hoặc pipeline sklearn/imblearn đã được fit.
        X: Dữ liệu đặc trưng.
        y: Nhãn thực tế.
        
    Returns:
        dict: Kết quả từ evaluate_model.
    """
    y_pred = pipeline.predict(X)
    
    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            # Lấy xác suất của lớp 1
            y_proba = pipeline.predict_proba(X)[:, 1]
        except Exception:
            pass
            
    return evaluate_model(y, y_pred, y_proba)

def plot_confusion_matrix(y_true, y_pred, title, save_path=None):
    """
    Vẽ Confusion Matrix dạng heatmap với số lượng và phần trăm.
    
    Args:
        y_true: Nhãn thực tế.
        y_pred: Nhãn dự đoán.
        title: Tiêu đề biểu đồ.
        save_path: Đường dẫn lưu biểu đồ (tùy chọn).
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_perc = cm / cm.sum() * 100
    
    labels = [f"{v1}\n({v2:.1f}%)" for v1, v2 in zip(cm.flatten(), cm_perc.flatten())]
    labels = np.asarray(labels).reshape(2, 2)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues',
                xticklabels=['No churn', 'Churn'], 
                yticklabels=['No churn', 'Churn'])
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

def metrics_table(results: dict) -> pd.DataFrame:
    """
    Gom kết quả của nhiều mô hình thành một DataFrame để so sánh.
    
    Args:
        results: Dictionary chứa kết quả dạng {tên_model: dict_metrics}
        
    Returns:
        DataFrame so sánh các metrics.
    """
    df = pd.DataFrame(results).T
    return df

# Kiểm thử nhanh (assert)
if __name__ == "__main__":
    # Test case: All zeros (Dummy classifier prediction)
    y_test_true = np.array([0, 0, 1, 1])
    y_test_pred = np.array([0, 0, 0, 0])
    
    res = evaluate_model(y_test_true, y_test_pred)
    assert res['precision'] == 0.0
    assert res['recall'] == 0.0
    assert res['accuracy'] == 0.5
    assert res['roc_auc'] is None
    print("All tests passed.")
