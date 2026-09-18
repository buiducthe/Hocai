# -*- coding: utf-8 -*-
"""
HỆ THỐNG DỰ BÁO GIAN LẬN BÁO CÁO TÀI CHÍNH
Sử dụng mô hình Logistic Regression, XGBoost & 8 biến Beneish M-Score[cite: 1]
Tác giả: Chuyên gia Lập trình Web & Khoa học Dữ liệu[cite: 1]
Deploy: Streamlit Cloud[cite: 1]
"""

import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
    roc_curve, auc, precision_recall_curve
)

# -----------------------------------------------------------------------------
# CẤU HÌNH TRANG STREAMLIT
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dự Báo Gian Lận BCTC | Beneish M-Score & ML Models",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS giao diện
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    .alert-danger-box {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 14px 18px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .alert-success-box {
        background-color: #F0FDF4;
        border-left: 5px solid #22C55E;
        padding: 14px 18px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CÁC HẰNG SỐ & DANH MỤC 8 CHỈ SỐ BENEISH
# -----------------------------------------------------------------------------
FEATURES = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "TATA", "LVGI"]
TARGET = "FRAUD_FLAG"

FEATURE_DESCRIPTIONS = {
    "DSRI": {
        "vn": "Số ngày thu tiền khách hàng (Days Sales in Receivables Index)",
        "desc": "Tỷ lệ số ngày phải thu khách hàng năm nay so với năm trước. DSRI > 1 cho thấy phải thu tăng nhanh hơn doanh thu, cảnh báo khả năng ghi nhận doanh thu ảo."
    },
    "GMI": {
        "vn": "Biên lợi nhuận gộp (Gross Margin Index)",
        "desc": "Tỷ lệ biên lợi nhuận gộp năm trước so với năm nay. GMI > 1 phản ánh biên lợi nhuận suy giảm, tạo động cơ thổi phồng doanh thu hoặc giảm chi phí."
    },
    "AQI": {
        "vn": "Chất lượng tài sản (Asset Quality Index)",
        "desc": "Tỷ lệ tài sản phi hiện hữu (ngoài TSCĐ, tiền mặt, tài sản lưu động) so với tổng tài sản. AQI > 1 biểu thị xu hướng vốn hóa chi phí hoặc tài sản kém chất lượng."
    },
    "SGI": {
        "vn": "Tăng trưởng doanh thu (Sales Growth Index)",
        "desc": "Tỷ lệ doanh thu năm nay so với năm trước. Tăng trưởng cao tạo áp lực duy trì kỳ vọng thị trường, tăng động cơ gian lận."
    },
    "DEPI": {
        "vn": "Tỷ lệ khấu hao (Depreciation Index)",
        "desc": "Tỷ lệ khấu hao năm trước so với năm nay. DEPI > 1 cho thấy DN kéo dài thời gian khấu hao hoặc đổi phương pháp để tăng lợi nhuận."
    },
    "SGAI": {
        "vn": "Chi phí bán hàng & QLDN (Sales, General & Admin Expense Index)",
        "desc": "Tỷ lệ chi phí SG&A trên doanh thu năm nay so với năm trước. SGAI > 1 cho thấy hiệu quả quản lý chi phí sụt giảm."
    },
    "TATA": {
        "vn": "Biến dồn tích trên tổng tài sản (Total Accruals to Total Assets)",
        "desc": "Chênh lệch giữa Lợi nhuận thuần và Dòng tiền thuần từ HĐKD chia cho Tổng tài sản. TATA càng cao phản ánh lợi nhuận ít được bảo chứng bằng dòng tiền thực."
    },
    "LVGI": {
        "vn": "Đòn bẩy tài chính (Leverage Index)",
        "desc": "Tỷ lệ nợ trên tổng tài sản năm nay so với năm trước. LVGI > 1 cho thấy rủi ro nợ vay tăng, gây áp lực vi phạm các cam kết tín dụng."
    }
}

# -----------------------------------------------------------------------------
# HÀM LOAD & XỬ LÝ DỮ LIỆU
# -----------------------------------------------------------------------------
@st.cache_data
def load_default_data():
    """Tự động tìm và đọc file dữ liệu mặc định MScore_data.csv"""[cite: 1]
    possible_paths = [
        "MScore_data.csv",
        os.path.join(os.path.dirname(__file__), "MScore_data.csv"),
        os.path.join(os.getcwd(), "MScore_data.csv")
    ]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                return df
            except Exception:
                pass
    return None

def clean_data(df):
    """Làm sạch và kiểm tra dữ liệu theo chuẩn mô hình"""[cite: 1]
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dữ liệu thiếu các cột bắt buộc: {missing_cols}")
    
    data = df[FEATURES + [TARGET]].copy()
    for c in FEATURES + [TARGET]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna().reset_index(drop=True)
    data[TARGET] = data[TARGET].astype(int)

    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Cột FRAUD_FLAG chỉ được chứa giá trị nhị phân 0 (Không gian lận) hoặc 1 (Gian lận).")
    
    return data

@st.cache_data
def train_model(data, model_type="Logistic Regression", test_size=0.20, random_state=42, n_estimators=100, max_depth=3, learning_rate=0.1):
    """Huấn luyện mô hình Pipeline: StandardScaler + LogisticRegression hoặc XGBoost"""[cite: 1]
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    if model_type == "Logistic Regression":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=random_state))
        ])
        pipeline.fit(X_train, y_train)
        y_test_prob = pipeline.predict_proba(X_test)[:, 1]

        model_obj = pipeline.named_steps["model"]
        intercept = float(model_obj.intercept_[0])
        coef = model_obj.coef_[0]

        coef_df = pd.DataFrame({
            "Chỉ số": FEATURES,
            "Hệ số (Beta)": coef,
            "Odds Ratio (e^Beta)": np.exp(coef),
            "Chiều tác động": ["Tăng nguy cơ gian lận" if b > 0 else "Giảm nguy cơ gian lận" for b in coef]
        })
        return pipeline, X_train, X_test, y_train, y_test, y_test_prob, intercept, coef, coef_df

    else: # XGBoost
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                eval_metric="logloss"
            ))
        ])
        pipeline.fit(X_train, y_train)
        y_test_prob = pipeline.predict_proba(X_test)[:, 1]

        model_obj = pipeline.named_steps["model"]
        importances = model_obj.feature_importances_

        coef_df = pd.DataFrame({
            "Chỉ số": FEATURES,
            "Mức độ quan trọng (Feature Importance)": importances
        }).sort_values(by="Mức độ quan trọng (Feature Importance)", ascending=False).reset_index(drop=True)

        return pipeline, X_train, X_test, y_train, y_test, y_test_prob, 0.0, importances, coef_df

def compute_metrics(y_true, y_prob, threshold=0.5):
    """Tính toán đầy đủ các chỉ số đánh giá theo ngưỡng phân loại threshold"""[cite: 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics_df = pd.DataFrame({
        "Chỉ tiêu đánh giá": [
            "Độ chính xác (Accuracy)",
            "Độ chuẩn xác (Precision)",
            "Độ nhạy / Thu hồi (Recall / Sensitivity)",
            "Điểm F1-Score",
            "Độ đặc hiệu (Specificity)",
            "Tỷ lệ dương tính giả (FPR)",
            "Tỷ lệ âm tính giả (FNR)"
        ],
        "Giá trị": [acc, prec, rec, f1, spec, fpr, fnr],
        "Tỷ lệ (%)": [acc * 100, prec * 100, rec * 100, f1 * 100, spec * 100, fpr * 100, fnr * 100]
    })

    return cm, tn, fp, fn, tp, metrics_df, y_pred

def calculate_beneish_mscore(dsri, gmi, aqi, sgi, depi, sgai, tata, lvgi):
    """Tính điểm Beneish M-Score nguyên bản (Beneish 1999 8-variable model)"""[cite: 1]
    m_score = (
        -4.84
        + 0.920 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.037 * tata
        + 0.0327 * lvgi
    )
    return m_score

# -----------------------------------------------------------------------------
# THANH ĐIỀU HƯỚNG BÊN TRÁI (SIDEBAR)
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/bullish.png", width=70)
st.sidebar.title("Cấu hình Hệ thống")

# 1. Nguồn dữ liệu huấn luyện
st.sidebar.subheader("1. Dữ liệu huấn luyện")
uploaded_file = st.sidebar.file_uploader("Tải lên file CSV dữ liệu mới (nếu có)", type=["csv"])[cite: 1]

raw_df = None
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Đã nạp file: {uploaded_file.name}")
    except Exception as e:
        st.sidebar.error(f"Lỗi khi đọc file: {e}")
else:
    raw_df = load_default_data()
    if raw_df is not None:
        st.sidebar.info("Đang sử dụng dữ liệu mặc định: `MScore_data.csv`")[cite: 1]
    else:
        st.sidebar.warning("Chưa có dữ liệu mặc định. Vui lòng tải lên file CSV!")

if raw_df is None:
    st.title("⚖️ HỆ THỐNG DỰ BÁO GIAN LẬN BÁO CÁO TÀI CHÍNH")
    st.error("⚠️ Không tìm thấy file dữ liệu `MScore_data.csv`. Vui lòng tải lên file CSV ở thanh bên trái (Sidebar) để tiếp tục!")[cite: 1]
    st.stop()

try:
    data = clean_data(raw_df)
except Exception as e:
    st.error(f"Lỗi định dạng dữ liệu: {e}")
    st.stop()

# 2. Chọn mô hình và tham số
st.sidebar.subheader("2. Thuật toán & Tham số Mô hình")
model_choice = st.sidebar.selectbox("Chọn mô hình Học máy", ["Logistic Regression", "XGBoost"])

n_estimators_val = 100
max_depth_val = 3
learning_rate_val = 0.1

if model_choice == "XGBoost":
    n_estimators_val = st.sidebar.slider("Số lượng cây (n_estimators)", 50, 300, 100, 25)
    max_depth_val = st.sidebar.slider("Độ sâu tối đa (max_depth)", 2, 8, 3, 1)
    learning_rate_val = st.sidebar.slider("Tốc độ học (learning_rate)", 0.01, 0.3, 0.1, 0.01)

test_size_ratio = st.sidebar.slider("Tỷ lệ tập kiểm tra (Test size)", min_value=0.10, max_value=0.40, value=0.20, step=0.05)[cite: 1]
random_seed = st.sidebar.number_input("Random Seed (để tái lập kết quả)", value=42, step=1)[cite: 1]

# 3. Ngưỡng quyết định (Classification Threshold)
st.sidebar.subheader("3. Ngưỡng phân loại (Threshold)")
threshold = st.sidebar.slider(
    "Ngưỡng xác suất phân loại gian lận",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.01,
    help="Xác suất P >= Ngưỡng sẽ được xếp loại là GIAN LẬN (1). Hạ thấp ngưỡng giúp tăng độ nhạy (Recall), bắt được nhiều ca gian lận hơn nhưng có thể tăng cảnh báo nhầm (FPR)."
)

# Huấn luyện mô hình
pipeline, X_train, X_test, y_train, y_test, y_test_prob, intercept, coef_raw, coef_df = train_model(
    data, model_type=model_choice, test_size=test_size_ratio, random_state=int(random_seed),
    n_estimators=n_estimators_val, max_depth=max_depth_val, learning_rate=learning_rate_val
)

# Đánh giá theo threshold hiện tại
cm, tn, fp, fn, tp, metrics_df, y_pred = compute_metrics(y_test, y_test_prob, threshold=threshold)

# -----------------------------------------------------------------------------
# TIÊU ĐỀ CHÍNH CỦA WEB APP
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">⚖️ HỆ THỐNG DỰ BÁO GIAN LẬN BÁO CÁO TÀI CHÍNH</div>', unsafe_allow_html=True)[cite: 1]
st.markdown(f'<div class="sub-header">Ứng dụng Học máy ({model_choice}) kết hợp 8 chỉ số Beneish M-Score giúp phát hiện sớm rủi ro thao túng BCTC</div>', unsafe_allow_html=True)[cite: 1]

# -----------------------------------------------------------------------------
# CÁC TAB CHỨC NĂNG
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Tổng quan Dữ liệu & EDA",
    "📈 Huấn luyện & Đánh giá Mô hình",
    "🔍 Dự báo Doanh nghiệp Đơn lẻ",
    "📁 Dự báo Hàng loạt (Batch)",
    "📖 Hướng dẫn & Lý thuyết Beneish"
])

# =============================================================================
# TAB 1: TỔNG QUAN DỮ LIỆU & EDA
# =============================================================================
with tab1:
    st.subheader("1. Tổng quan Bộ dữ liệu")[cite: 1]
    
    total_samples = len(data)
    fraud_count = int((data[TARGET] == 1).sum())
    non_fraud_count = int((data[TARGET] == 0).sum())
    fraud_rate = (fraud_count / total_samples) * 100

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Tổng số quan sát</div>
            <div class="metric-value">{total_samples:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Không gian lận (Nhãn 0)</div>
            <div class="metric-value" style="color: #10B981;">{non_fraud_count:,} ({100-fraud_rate:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Gian lận (Nhãn 1)</div>
            <div class="metric-value" style="color: #EF4444;">{fraud_count:,} ({fraud_rate:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Số biến dự báo</div>
            <div class="metric-value" style="color: #3B82F6;">{len(FEATURES)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")[cite: 1]
    
    col_g1, col_g2 = st.columns([1, 1])
    with col_g1:
        st.markdown("##### 📌 Tỷ lệ phân bố Nhãn Gian lận (FRAUD_FLAG)")
        pie_fig = px.pie(
            values=[non_fraud_count, fraud_count],
            names=["Không gian lận (0)", "Gian lận (1)"],
            color=["Không gian lận (0)", "Gian lận (1)"],
            color_discrete_map={"Không gian lận (0)": "#10B981", "Gian lận (1)": "#EF4444"},
            hole=0.45
        )
        pie_fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=320)
        st.plotly_chart(pie_fig, use_container_width=True)
        
    with col_g2:
        st.markdown("##### 📌 Ma trận Tương quan (Correlation Heatmap)")
        corr = data[FEATURES + [TARGET]].corr()
        heat_fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            origin="lower"
        )
        heat_fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=320)
        st.plotly_chart(heat_fig, use_container_width=True)

    st.markdown("##### 📋 Bảng Dữ liệu mẫu & Thống kê mô tả")
    col_t1, col_t2 = st.columns([1.2, 0.8])
    with col_t1:
        st.write("**Xem trước dữ liệu (10 dòng đầu tiên):**")
        st.dataframe(data.head(10), use_container_width=True)
    with col_t2:
        st.write("**Thống kê mô tả các biến:**")
        st.dataframe(data[FEATURES].describe().T[["mean", "std", "min", "50%", "max"]].rename(columns={"50%": "median"}), use_container_width=True)

    st.markdown("##### 📊 So sánh Phân phối từng chỉ số Beneish giữa 2 nhóm")
    selected_feature = st.selectbox("Chọn chỉ số Beneish để xem phân phối:", FEATURES)
    
    col_box, col_hist = st.columns(2)
    with col_box:
        box_fig = px.box(
            data,
            x=TARGET,
            y=selected_feature,
            color=TARGET,
            color_discrete_map={0: "#10B981", 1: "#EF4444"},
            labels={TARGET: "Gian lận (0 = Không, 1 = Có)", selected_feature: f"Giá trị {selected_feature}"},
            title=f"Boxplot so sánh {selected_feature} theo Nhãn"
        )
        box_fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(box_fig, use_container_width=True)
        
    with col_hist:
        hist_fig = px.histogram(
            data,
            x=selected_feature,
            color=TARGET,
            barmode="overlay",
            color_discrete_map={0: "#10B981", 1: "#EF4444"},
            labels={TARGET: "Gian lận", selected_feature: f"Giá trị {selected_feature}"},
            title=f"Histogram phân phối {selected_feature}"
        )
        hist_fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(hist_fig, use_container_width=True)

# =============================================================================
# TAB 2: HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH
# =============================================================================
with tab2:
    st.subheader(f"2. Kết quả Huấn luyện & Đánh giá Mô hình ({model_choice})")
    st.caption(f"Mô hình được huấn luyện trên {len(X_train)} mẫu (Train: {100-test_size_ratio*100:.0f}%) và đánh giá trên {len(X_test)} mẫu (Test: {test_size_ratio*100:.0f}%). Ngưỡng phân loại hiện tại: **{threshold:.2f}**")

    # Hiển thị các chỉ tiêu chính
    c_acc, c_rec, c_prec, c_f1, c_spec = st.columns(5)
    c_acc.metric("Độ chính xác (Accuracy)", f"{metrics_df.loc[0, 'Tỷ lệ (%)']:.2f}%")
    c_rec.metric("Độ nhạy (Recall/Sensitivity)", f"{metrics_df.loc[2, 'Tỷ lệ (%)']:.2f}%", help="Khả năng phát hiện đúng các công ty gian lận")
    c_prec.metric("Độ chuẩn xác (Precision)", f"{metrics_df.loc[1, 'Tỷ lệ (%)']:.2f}%", help="Tỷ lệ dự báo gian lận thực sự chính xác")
    c_f1.metric("F1-Score", f"{metrics_df.loc[3, 'Giá trị']:.4f}", help="Trung bình điều hòa giữa Precision và Recall")
    c_spec.metric("Độ đặc hiệu (Specificity)", f"{metrics_df.loc[4, 'Tỷ lệ (%)']:.2f}%", help="Khả năng nhận diện đúng các công ty an toàn")

    st.write("")[cite: 1]
    
    col_eval1, col_eval2 = st.columns([1, 1])
    
    with col_eval1:
        st.markdown("##### 🔲 Ma trận Nhầm lẫn (Confusion Matrix)")
        cm_labels_x = ["Dự báo: Không gian lận (0)", "Dự báo: Gian lận (1)"]
        cm_labels_y = ["Thực tế: Không gian lận (0)", "Thực tế: Gian lận (1)"]
        
        cm_text = [
            [f"Đúng âm tính (TN)<br><b>{tn}</b>", f"Dương tính giả (FP)<br><b>{fp}</b>"],
            [f"Âm tính giả (FN)<br><b>{fn}</b>", f"Đúng dương tính (TP)<br><b>{tp}</b>"]
        ]
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=cm_labels_x,
            y=cm_labels_y,
            text=cm_text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False
        ))
        fig_cm.update_layout(
            height=340,
            margin=dict(l=40, r=40, t=30, b=40),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.info(f"💡 **Phân tích:** Đúng âm tính (TN) = {tn} | Báo động nhầm (FP) = {fp} | Bỏ sót gian lận (FN) = {fn} | Bắt trúng gian lận (TP) = {tp}")

    with col_eval2:
        st.markdown("##### 📈 Đường cong ROC & ROC-AUC")
        fpr_arr, tpr_arr, _ = roc_curve(y_test, y_test_prob)
        roc_auc_val = auc(fpr_arr, tpr_arr)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_arr, y=tpr_arr, mode="lines", name=f"ROC (AUC = {roc_auc_val:.4f})", line=dict(color="#2563EB", width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ngẫu nhiên", line=dict(color="#9CA3AF", dash="dash")))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate (FPR)",
            yaxis_title="True Positive Rate (Recall / TPR)",
            height=340,
            margin=dict(l=40, r=40, t=30, b=40),
            legend=dict(x=0.55, y=0.1)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.write("---")[cite: 1]

    if model_choice == "Logistic Regression":
        st.markdown("##### 📐 Phương trình Hồi quy Logistic (Trên dữ liệu đã chuẩn hóa)")
        equation_str = f"$$\\text{{logit}}(P(\\text{{FRAUD}}=1)) = {intercept:.4f}"
        for name, b in zip(FEATURES, coef_raw):
            sign = "+" if b >= 0 else "-"
            equation_str += f" {sign} {abs(b):.4f} \\times \\text{{{name}}}_{{std}}"
        equation_str += "$$"
        st.markdown(equation_str)

        col_c1, col_c2 = st.columns([1.2, 0.8])
        with col_c1:
            st.markdown("##### 📊 Bảng Hệ số & Tỷ số Chênh (Odds Ratio)")
            st.dataframe(coef_df.style.format({
                "Hệ số (Beta)": "{:.4f}",
                "Odds Ratio (e^Beta)": "{:.4f}"
            }), use_container_width=True)

        with col_c2:
            st.markdown("##### 📑 Bảng Chi tiết các Chỉ tiêu Đo lường")
            st.dataframe(metrics_df.style.format({
                "Giá trị": "{:.4f}",
                "Tỷ lệ (%)": "{:.2f}%"
            }), use_container_width=True)
            
        interpretation_list = []
        for name, b in zip(FEATURES, coef_raw):
            or_val = np.exp(b)
            if b > 0:
                meaning = f"{name} tăng 1 độ lệch chuẩn làm tăng log-odds gian lận {b:.4f}; odds gian lận được nhân {or_val:.4f}."
            else:
                meaning = f"{name} tăng 1 độ lệch chuẩn làm giảm log-odds gian lận {abs(b):.4f}; odds gian lận được nhân {or_val:.4f}."
            interpretation_list.append({
                "Chỉ số": name, "Hệ số": b, "Odds Ratio": or_val, "Ý nghĩa kinh tế": meaning
            })
        interp_df = pd.DataFrame(interpretation_list)
        
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            coef_df.to_excel(writer, sheet_name="Coefficients", index=False)
            pd.DataFrame(cm, index=cm_labels_y, columns=cm_labels_x).to_excel(writer, sheet_name="Confusion_Matrix")
            metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
            interp_df.to_excel(writer, sheet_name="Interpretation", index=False)
        excel_buffer.seek(0)
        
        st.markdown("##### 📥 Xuất Toàn Bộ Kết Quả Ra File Excel")
        st.download_button(
            label="📥 Tải Báo Cáo Đầy Đủ (Excel .xlsx)",
            data=excel_buffer,
            file_name="Bao_Cao_Logistic_Regression_MScore.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    else: # XGBoost
        st.markdown("##### 🌲 Biểu đồ Mức độ Quan trọng của Biến (Feature Importance - XGBoost)")
        fig_imp = px.bar(
            coef_df,
            x="Mức độ quan trọng (Feature Importance)",
            y="Chỉ số",
            orientation="h",
            color="Mức độ quan trọng (Feature Importance)",
            color_continuous_scale="Viridis",
            title="Độ quan trọng của 8 chỉ số Beneish trong mô hình XGBoost"
        )
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}, height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_imp, use_container_width=True)

        col_x1, col_x2 = st.columns([1.2, 0.8])
        with col_x1:
            st.markdown("##### 📊 Bảng Mức độ Quan trọng các Biến")
            st.dataframe(coef_df.style.format({
                "Mức độ quan trọng (Feature Importance)": "{:.4f}"
            }), use_container_width=True)
        with col_x2:
            st.markdown("##### 📑 Bảng Chi tiết các Chỉ tiêu Đo lường")
            st.dataframe(metrics_df.style.format({
                "Giá trị": "{:.4f}",
                "Tỷ lệ (%)": "{:.2f}%"
            }), use_container_width=True)

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            coef_df.to_excel(writer, sheet_name="Feature_Importance", index=False)
            pd.DataFrame(cm, index=cm_labels_y, columns=cm_labels_x).to_excel(writer, sheet_name="Confusion_Matrix")
            metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
        excel_buffer.seek(0)
        
        st.markdown("##### 📥 Xuất Toàn Bộ Kết Quả Ra File Excel")
        st.download_button(
            label="📥 Tải Báo Cáo Đầy Đủ (Excel .xlsx)",
            data=excel_buffer,
            file_name="Bao_Cao_XGBoost_MScore.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# =============================================================================
# TAB 3: DỰ BÁO DOANH NGHIỆP ĐƠN LẺ
# =============================================================================
with tab3:
    st.subheader(f"3. Dự báo Rủi ro Gian lận cho Doanh nghiệp Cụ thể ({model_choice})")
    st.caption("Nhập giá trị 8 chỉ số Beneish để hệ thống tính toán xác suất gian lận và so sánh với mô hình M-Score nguyên bản.")

    st.markdown("###### ⚡ Chọn nạp nhanh cấu hình mẫu:")
    col_s1, col_s2, col_s3 = st.columns(3)
    
    default_vals = {
        "DSRI": 1.05, "GMI": 1.02, "AQI": 0.95, "SGI": 1.10,
        "DEPI": 0.98, "SGAI": 1.01, "TATA": 0.03, "LVGI": 1.02
    }
    
    if col_s1.button("🟢 Mẫu Doanh nghiệp An toàn (Bình thường)"):
        st.session_state["val_DSRI"] = 0.85
        st.session_state["val_GMI"] = 0.92
        st.session_state["val_AQI"] = 0.80
        st.session_state["val_SGI"] = 1.02
        st.session_state["val_DEPI"] = 1.00
        st.session_state["val_SGAI"] = 0.95
        st.session_state["val_TATA"] = -0.04
        st.session_state["val_LVGI"] = 0.90

    if col_s2.button("🚨 Mẫu Doanh nghiệp Rủi ro Gian lận Cao"):
        st.session_state["val_DSRI"] = 1.85
        st.session_state["val_GMI"] = 1.60
        st.session_state["val_AQI"] = 1.45
        st.session_state["val_SGI"] = 1.80
        st.session_state["val_DEPI"] = 1.25
        st.session_state["val_SGAI"] = 1.30
        st.session_state["val_TATA"] = 0.18
        st.session_state["val_LVGI"] = 1.35

    if col_s3.button("📊 Mẫu Trung bình Toàn bộ Dữ liệu"):
        means = data[FEATURES].mean()
        for f in FEATURES:
            st.session_state[f"val_{f}"] = float(means[f])

    st.write("")[cite: 1]
    with st.form("single_prediction_form"):
        st.markdown("##### 📝 Nhập 8 Chỉ số Tài chính Beneish:")
        c1, c2 = st.columns(2)
        
        with c1:
            in_dsri = st.number_input(
                f"1. DSRI - {FEATURE_DESCRIPTIONS['DSRI']['vn']}",
                value=float(st.session_state.get("val_DSRI", default_vals["DSRI"])),
                step=0.01, format="%.4f",
                help=FEATURE_DESCRIPTIONS["DSRI"]["desc"]
            )
            in_gmi = st.number_input(
                f"2. GMI - {FEATURE_DESCRIPTIONS['GMI']['vn']}",
                value=float(st.session_state.get("val_GMI", default_vals["GMI"])),
                step=0.01, format="%.4f",
                help=FEATURE_DESCRIPTIONS["GMI"]["desc"]
            )
            in_aqi = st.number_input(
                f"3. AQI - {FEATURE_DESCRIPTIONS['AQI']['vn']}",
                value=float(st.session_state.get("val_AQI", default_vals["AQI"])),
                step=0.01, format="%.4f",
                help=FEATURE_DESCRIPTIONS["AQI"]["desc"]
            )
            in_sgi = st.number_input(
                f"4. SGI - {FEATURE_DESCRIPTIONS['SGI']['vn']}",
                value=float(st.session_state.get("val_SGI", default_vals["SGI"])),
                step=0.01, format="%.4f",
                help=FEATURE_DESCRIPTIONS["SGI"]["desc"]
            )

        with c2:
            in_depi = st.number_input(
                f"5. DEPI - {FEATURE_DESCRIPTIONS['DEPI']['vn']}",
                value=float(st.session_state.get("val_DEPI", default_vals["DEPI"])),
                step=0.01, format="%.4f",
                help=FEATURE_DESCRIPTIONS["DEPI"]["desc"]
            )
            in_sgai = st.number_input(
                f"6. SGAI - {FEATURE_DESCRIPTIONS['SGAI']['vn']}",
                value=float(st.session_state.get("val_SGAI", default_vals["SGAI"])),
                step=0.01, format="%.4f",
                help=FEATURE_DESCRIPTIONS["SGAI"]["desc"]
            )
            in_tata = st.number_input(
                f"7. TATA - {FEATURE_DESCRIPTIONS['TATA']['vn']}",
                value=float(st.session_state.get("val_TATA", default_vals["TATA"])),
                step=0.01, format="%.4f",
                help=FEATURE_DESCRIPTIONS["TATA"]["desc"]
            )
            in_lvgi = st.number_input(
                f"8. LVGI - {FEATURE_DESCRIPTIONS['LVGI']['vn']}",
                value=float(st.session_state.get("val_LVGI", default_vals["LVGI"])),
                step=0.01, format="%.4f",
                help=FEATURE_DESCRIPTIONS["LVGI"]["desc"]
            )

        submit_single = st.form_submit_button("🚀 Thực hiện Dự báo Gian lận", use_container_width=True)

    if submit_single:
        input_dict = {
            "DSRI": [in_dsri], "GMI": [in_gmi], "AQI": [in_aqi], "SGI": [in_sgi],
            "DEPI": [in_depi], "SGAI": [in_sgai], "TATA": [in_tata], "LVGI": [in_lvgi]
        }
        input_df = pd.DataFrame(input_dict)

        fraud_proba = float(pipeline.predict_proba(input_df)[0, 1])
        is_fraud = fraud_proba >= threshold

        m_score_val = calculate_beneish_mscore(
            in_dsri, in_gmi, in_aqi, in_sgi, in_depi, in_sgai, in_tata, in_lvgi
        )
        is_beneish_manipulator = m_score_val > -2.22

        st.write("---")[cite: 1]
        st.markdown("#### 🎯 Kết quả Phân tích & Dự báo")

        col_res1, col_res2 = st.columns([1.1, 0.9])

        with col_res1:
            if is_fraud:
                st.markdown(f"""
                <div class="alert-danger-box">
                    <h3 style="color: #B91C1C; margin:0 0 8px 0;">🚨 CẢNH BÁO: NGUY CƠ GIAN LẬN CAO</h3>
                    <p style="margin:0; font-size: 1.05rem;">
                        Mô hình <b>{model_choice}</b> ước tính xác suất gian lận BCTC của doanh nghiệp này là 
                        <b>{fraud_proba*100:.2f}%</b> (vượt qua ngưỡng phân loại <b>{threshold*100:.1f}%</b>).
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-success-box">
                    <h3 style="color: #15803D; margin:0 0 8px 0;">🟢 AN TOÀN: NGUY CƠ GIAN LẬN THẤP</h3>
                    <p style="margin:0; font-size: 1.05rem;">
                        Mô hình <b>{model_choice}</b> ước tính xác suất gian lận BCTC của doanh nghiệp này là 
                        <b>{fraud_proba*100:.2f}%</b> (thấp hơn ngưỡng phân loại <b>{threshold*100:.1f}%</b>).
                    </p>
                </div>
                """, unsafe_allow_html=True)

            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fraud_proba * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Xác suất Gian lận P(Fraud) - {model_choice} (%)", 'font': {'size': 16}},
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#DC2626" if is_fraud else "#16A34A"},
                    'steps': [
                        {'range': [0, threshold * 100], 'color': "#DCFCE7"},
                        {'range': [threshold * 100, 100], 'color': "#FEE2E2"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': threshold * 100
                    }
                }
            ))
            gauge_fig.update_layout(height=260, margin=dict(l=30, r=30, t=40, b=20))
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col_res2:
            st.markdown("##### 🏛️ So sánh với Mô hình Beneish M-Score gốc (1999)")
            st.markdown(f"""
            - **Giá trị Beneish M-Score**: `M = {m_score_val:.4f}`
            - **Ngưỡng chuẩn**: `M > -2.22` biểu thị khả năng gian lận cao.
            - **Kết luận Beneish gốc**: {"⚠️ **Có dấu hiệu thao túng (Manipulator)**" if is_beneish_manipulator else "✅ **Không có dấu hiệu thao túng (Non-manipulator)**"}
            """)
            
            if is_fraud == is_beneish_manipulator:
                st.success("✅ **Sự đồng thuận cao:** Cả mô hình học máy và Beneish M-Score đều đưa ra kết luận thống nhất!")
            else:
                st.warning("⚠️ **Lưu ý khác biệt:** Có sự khác biệt giữa mô hình học máy chọn lựa và công thức gốc năm 1999.")

            st.markdown("##### 📊 So sánh với Trung bình Dữ liệu Huấn luyện:")
            means = data[FEATURES].mean()
            comparison_df = pd.DataFrame({
                "Chỉ số": FEATURES,
                "Doanh nghiệp này": [input_dict[f][0] for f in FEATURES],
                "Trung bình mẫu": [means[f] for f in FEATURES]
            })
            
            bar_comp = px.bar(
                comparison_df,
                x="Chỉ số",
                y=["Doanh nghiệp này", "Trung bình mẫu"],
                barmode="group",
                color_discrete_sequence=["#2563EB", "#94A3B8"]
            )
            bar_comp.update_layout(height=240, margin=dict(l=20, r=20, t=10, b=20), legend=dict(orientation="h", y=1.1))
            st.plotly_chart(bar_comp, use_container_width=True)

# =============================================================================
# TAB 4: DỰ BÁO HÀNG LOẠT (BATCH PREDICTION)
# =============================================================================
with tab4:
    st.subheader(f"4. Dự báo Rủi ro Gian lận Hàng loạt từ File CSV / Excel ({model_choice})")
    st.caption("Tải lên danh sách gồm nhiều doanh nghiệp để hệ thống tự động quét và phân loại rủi ro gian lận hàng loạt.")

    template_df = pd.DataFrame([
        {"COMPANY_NAME": "Công ty A", "DSRI": 1.05, "GMI": 0.98, "AQI": 0.85, "SGI": 1.12, "DEPI": 1.01, "SGAI": 0.97, "TATA": 0.02, "LVGI": 1.05},
        {"COMPANY_NAME": "Công ty B", "DSRI": 1.72, "GMI": 1.45, "AQI": 1.30, "SGI": 1.65, "DEPI": 1.15, "SGAI": 1.20, "TATA": 0.15, "LVGI": 1.28},
        {"COMPANY_NAME": "Công ty C", "DSRI": 0.92, "GMI": 1.03, "AQI": 0.78, "SGI": 0.95, "DEPI": 0.92, "SGAI": 1.02, "TATA": -0.05, "LVGI": 0.88}
    ])
    template_csv = template_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📄 Tải File Template Mẫu (.csv)",
        data=template_csv,
        file_name="template_du_bao_gian_lan.csv",
        mime="text/csv"
    )

    st.write("")[cite: 1]
    batch_file = st.file_uploader("Tải lên file danh sách công ty (.csv hoặc .xlsx)", type=["csv", "xlsx", "xls"], key="batch_uploader")

    if batch_file is not None:
        try:
            if batch_file.name.endswith(".csv"):
                batch_df = pd.read_csv(batch_file)
            else:
                batch_df = pd.read_excel(batch_file)
                
            st.success(f"Đã nạp file thành công! Tổng số dòng: {len(batch_df)}")
            
            missing_batch_cols = [c for c in FEATURES if c not in batch_df.columns]
            if missing_batch_cols:
                st.error(f"File tải lên thiếu các cột bắt buộc: {missing_batch_cols}")
            else:
                pred_input = batch_df[FEATURES].copy()
                for c in FEATURES:
                    pred_input[c] = pd.to_numeric(pred_input[c], errors="coerce").fillna(0)

                batch_probs = pipeline.predict_proba(pred_input)[:, 1]
                batch_preds = (batch_probs >= threshold).astype(int)

                batch_mscores = [
                    calculate_beneish_mscore(
                        row["DSRI"], row["GMI"], row["AQI"], row["SGI"],
                        row["DEPI"], row["SGAI"], row["TATA"], row["LVGI"]
                    ) for _, row in pred_input.iterrows()
                ]

                result_df = batch_df.copy()
                result_df["Xac_Suat_Gian_Lan"] = np.round(batch_probs, 4)
                result_df["Xac_Suat_Phan_Tram"] = np.round(batch_probs * 100, 2)
                result_df["Du_Bao_Gian_Lan"] = batch_preds
                result_df["Ket_Luan"] = ["🚨 Nguy cơ cao" if p == 1 else "🟢 An toàn" for p in batch_preds]
                result_df["Beneish_MScore"] = np.round(batch_mscores, 4)
                result_df["Beneish_Canh_Bao"] = ["Có thao túng" if m > -2.22 else "Bình thường" for m in batch_mscores]

                num_flagged = int((batch_preds == 1).sum())
                num_safe = int((batch_preds == 0).sum())

                st.write("")[cite: 1]
                col_b1, col_b2, col_b3 = st.columns(3)
                col_b1.metric("Tổng số công ty quét", f"{len(batch_df):,}")
                col_b2.metric("Số công ty CẢNH BÁO GIAN LẬN", f"{num_flagged:,}", f"{(num_flagged/len(batch_df))*100:.1f}%", delta_color="inverse")
                col_b3.metric("Số công ty AN TOÀN", f"{num_safe:,}", f"{(num_safe/len(batch_df))*100:.1f}%")

                st.markdown("##### 📋 Bảng Chi tiết Kết quả Dự báo:")
                st.dataframe(
                    result_df.style.apply(
                        lambda row: ['background-color: #FEE2E2' if row["Du_Bao_Gian_Lan"] == 1 else 'background-color: #ECFDF5' for _ in row],
                        axis=1
                    ),
                    use_container_width=True
                )

                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    csv_res = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Tải Kết Quả (CSV)",
                        data=csv_res,
                        file_name="Ket_qua_du_bao_hang_loat.csv",
                        mime="text/csv"
                    )
                with col_d2:
                    excel_res_buf = io.BytesIO()
                    result_df.to_excel(excel_res_buf, index=False, engine="openpyxl")
                    excel_res_buf.seek(0)
                    st.download_button(
                        label="📥 Tải Kết Quả (Excel .xlsx)",
                        data=excel_res_buf,
                        file_name="Ket_qua_du_bao_hang_loat.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

        except Exception as err:
            st.error(f"Lỗi khi xử lý file hàng loạt: {err}")

# =============================================================================
# TAB 5: HƯỚNG DẪN & LÝ THUYẾT BENEISH
# =============================================================================
with tab5:
    st.subheader("5. Cơ sở Lý thuyết & Hướng dẫn Sử dụng Hệ thống")[cite: 1]
    
    st.markdown("""
    ### 🏛️ Mô hình Beneish M-Score & Các Mô hình Học máy
    Mô hình **Beneish M-Score** được Giáo sư Messod Beneish phát triển vào năm 1999[cite: 1]. 
    Hệ thống hiện đại tích hợp cả hai thuật toán **Logistic Regression** (hồi quy tuyến tính xác suất) và **XGBoost** (mô hình học máy cây quyết định tăng cường gradient mạnh mẽ) để tối ưu hóa khả năng nhận diện các mẫu hình phi tuyến tính và phức tạp trong thao túng báo cáo tài chính.

    ---

    ### 📌 Chi tiết 8 Chỉ số Tài chính Beneish:
    """)

    for k, v in FEATURE_DESCRIPTIONS.items():
        st.markdown(f"""
        - **`{k}` - {v['vn']}**:
          {v['desc']}
        """)

    st.markdown("""
    ---

    ### ⚙️ Phương trình Mô hình Beneish gốc (1999):
    $$\\text{M-Score} = -4.84 + 0.920 \\times \\text{DSRI} + 0.528 \\times \\text{GMI} + 0.404 \\times \\text{AQI} + 0.892 \\times \\text{SGI} + 0.115 \\times \\text{DEPI} - 0.172 \\times \\text{SGAI} + 4.037 \\times \\text{TATA} + 0.0327 \\times \\text{LVGI}$$
    
    > **Quy tắc phân loại gốc:**
    > - Nếu $\\text{M-Score} > -2.22$: Doanh nghiệp có xác suất cao đang thao túng lợi nhuận.
    > - Nếu $\\text{M-Score} \\le -2.22$: Doanh nghiệp không có dấu hiệu thao túng bất thường.

    ---

    ### 🤖 Ưu thế của việc tích hợp XGBoost & Logistic Regression:
    1. **Đa dạng hóa mô hình:** Cho phép người dùng chuyển đổi linh hoạt giữa Logistic Regression (dễ diễn giải hệ số Odds Ratio) và XGBoost (bắt tốt các tương tác phi tuyến tính giữa các chỉ số tài chính).
    2. **Xác suất mềm (Calibrated Probabilities):** Cung cấp xác suất rủi ro cụ thể $P(\\text{Fraud}) \\in [0, 1]$.
    3. **Tùy biến ngưỡng quyết định (Flexible Threshold):** Giúp kiểm toán viên điều chỉnh linh hoạt theo khẩu vị rủi ro.
    """)

# -----------------------------------------------------------------------------
# PHẦN CHÂN TRANG (FOOTER)
# -----------------------------------------------------------------------------
st.write("")[cite: 1]
st.write("---")[cite: 1]
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.88rem;">
    ⚖️ <b>Hệ thống Dự báo Gian lận Báo cáo Tài chính</b> | Xây dựng trên nền tảng Streamlit, Scikit-Learn & XGBoost[cite: 1].<br>
    Ứng dụng được thiết kế tối ưu cho kiểm toán viên, nhà phân tích tài chính và nhà đầu tư.
</div>
""", unsafe_allow_html=True)