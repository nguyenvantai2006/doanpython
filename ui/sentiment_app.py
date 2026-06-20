"""
sentiment_app.py — Vietnamese Sentiment Analysis Streamlit UI
=========================================================================
A feature-rich interface for sentiment analysis using Streamlit.

Run with:
    streamlit run ui/sentiment_app.py
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Optional, List, Dict
import io, os

# ── Configuration ─────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_URL}/predict"
BATCH_ENDPOINT = f"{API_URL}/predict/batch"
INFO_ENDPOINT = f"{API_URL}/"

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Phân tích cảm xúc ngôn ngữ tiếng Việt",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session State Initialization ──────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "trend_data" not in st.session_state:
    st.session_state.trend_data = []
if "api_url" not in st.session_state:
    st.session_state.api_url = API_URL
if "single_text" not in st.session_state:
    st.session_state.single_text = ""
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "bert"
if "selected_sub_model" not in st.session_state:
    st.session_state.selected_sub_model = "demo"

# ── Example Templates ─────────────────────────────────────────────
EXAMPLES = {
    "Positive": "chất lượng tuyệt vời giao hàng rất nhanh shop tuyệt vời",
    "Neutral": "sản phẩm bình thường giá chấp nhận được",
    "Negative": "hàng kém chất lượng giao sai địa chỉ không đáng tiền"
}

# ── Helper Functions ──────────────────────────────────────────────
def get_model_info():
    """Fetch model performance metrics from API."""
    try:
        response = requests.get(st.session_state.api_url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def predict_single(text: str) -> Optional[Dict]:
    """Send single text to API for sentiment analysis with dynamic model choice."""
    try:
        # Cải tiến: Lấy mô hình được chọn động từ giao diện gửi đi
        payload = {
            "text": text, 
            "model": st.session_state.selected_model,
            "sub_model": st.session_state.selected_sub_model if st.session_state.selected_model == "bert" else "demo"
        }
        response = requests.post(
            f"{st.session_state.api_url}/predict",
            json=payload,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Lỗi từ API: {response.json().get('detail', response.text)}")
    except Exception as e:
        st.error(f"Lỗi kết nối: {str(e)}")
    return None

def predict_batch(texts: List[str]) -> Optional[Dict]:
    """Send batch texts to API for sentiment analysis with dynamic model choice."""
    try:
        # Cải tiến: Lấy mô hình được chọn động từ giao diện gửi đi phục vụ phân tích hàng loạt
        payload = {
            "texts": texts, 
            "model": st.session_state.selected_model,
            "sub_model": st.session_state.selected_sub_model if st.session_state.selected_model == "bert" else "demo"
        }
        response = requests.post(
            f"{st.session_state.api_url}/predict/batch",
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Lỗi từ API: {response.json().get('detail', response.text)}")
    except Exception as e:
        st.error(f"Lỗi kết nối: {str(e)}")
    return None

def add_to_history(text: str, result: Dict):
    """Add analysis to history and trend data."""
    entry = {
        "timestamp": datetime.now(),
        "text": text,
        "sentiment": result.get("sentiment", "Unknown"),
        "label": result.get("label", "Unknown"),
        "confidence": result.get("confidence", 0),
        "keywords": result.get("keywords", [])
    }
    st.session_state.history.append(entry)
    st.session_state.trend_data.append({
        "timestamp": entry["timestamp"],
        "label": entry["label"],
        "confidence": entry["confidence"]
    })

def export_to_csv(data: List[Dict]) -> bytes:
    """Convert analysis data to CSV."""
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode()

def set_example_text(example_text: str):
    """Callback to set example text in session state."""
    st.session_state.single_text = example_text

# ── Header ────────────────────────────────────────────────────────
st.title("Phân tích cảm xúc ngôn ngữ tiếng Việt")
st.markdown("Phân tích nhận xét sản phẩm tiếng Việt")
st.markdown("Nhập văn bản tiếng Việt để phân tích cảm xúc: POSITIVE, NEUTRAL, NEGATIVE")

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Cài đặt")
    
    # API Configuration
    st.session_state.api_url = st.text_input("API URL:", value=st.session_state.api_url)
    
    # Test Connection
    if st.button("Kiểm tra kết nối", width="stretch"):
        try:
            response = requests.get(f"{st.session_state.api_url}/health", timeout=30)
            if response.status_code == 200:
                st.success("API trực tuyến và hoạt động tốt!")
            else:
                st.error("API trả về lỗi. Vui lòng kiểm tra lại.")
        except Exception as e:
            st.error(f"Cannot reach API: {e}")
    
    st.divider()
    # Model Selection Info From API
    model_info = get_model_info()
    bert_available = model_info and model_info.get("bert_available", False)
    available_bert_models = model_info.get("available_bert_models", []) if model_info else []
    bert_models_info = model_info.get("bert_models_info", []) if model_info else []
    
    # 1. Định nghĩa danh sách các mô hình chính khớp với API
    MODEL_OPTIONS = {
        "bert": "TinyBert",
        "bilstm": "BiLSTM"
    }

    # 2. Thanh chọn mô hình chính
    st.selectbox(
        "Chọn mô hình phân tích cảm xúc:",
        options=list(MODEL_OPTIONS.keys()),
        format_func=lambda x: MODEL_OPTIONS.get(x, x),
        key="selected_model"
    )

    # 3. Nhánh hiển thị sub-model bổ sung (Chỉ hiện khi chọn BERT)
    if st.session_state.selected_model == "bert" and available_bert_models:
        sub_model_labels = {
            "phoBert": "PhoBERT (Vietnamese)",
            "mBert": "mBERT (Multilingual)",
            "demo": "Demo (Tiny BERT - 76%)",
            "legacy": "Legacy (Default)",
        }
        sub_model_options = available_bert_models
        
        st.selectbox(
            "Chọn BERT sub-model:",
            options=sub_model_options,
            format_func=lambda x: sub_model_labels.get(x, x),
            key="selected_sub_model"
        )
    else:
        st.session_state.selected_sub_model = "demo"
        
    st.divider()
    
   
   
   # Model Info Dynamic Cards
    st.markdown("### Thông tin Mô hình")
    
    if st.session_state.selected_model == "ml":
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "73.37%")
        with col2:
            st.metric("F1 Score", "63.81%")
        st.caption("Mô hình: Machine Learning - TF-IDF (Mô hình truyền thống)")
        
    elif st.session_state.selected_model == "bilstm":
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "80.04%")
        with col2:
            st.metric("F1 Score", "79.65%")
        st.caption("Mô hình: Mạng nơ-ron chuỗi - BiLSTM (Mô hình đối chứng)")
        
    elif st.session_state.selected_model == "bert":
        # Chốt điểm chuẩn của BERT để hiển thị demo hoàn chỉnh
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "76.23%")
        with col2:
            st.metric("F1 Score", "64.95%")
        st.caption("Mô hình: Mạng nơ-ron tiên tiến - BERT (Mô hình đề xuất)")
            
    st.divider()
   
    if st.session_state.history:
        st.divider()
        st.markdown("### Thống kê phiên làm việc")
        history_df = pd.DataFrame(st.session_state.history)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tổng số", len(st.session_state.history))
        with col2:
            positive = (history_df["label"] == "POSITIVE").sum()
            st.metric("Tích cực", positive)
        with col3:
            negative = (history_df["label"] == "NEGATIVE").sum()
            st.metric("Tiêu cực", negative)

# ── Main Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Phân tích đơn lẻ",
    "Phân tích hàng loạt",
    "Thống kê tổng quan",
    "Lịch sử hoạt động"
])

# ══════════════════════════════════════════════════════════════════
# TAB 1: Single Analysis
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Phân tích văn bản đơn lẻ")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_text = st.text_area(
            "Nhập văn bản tiếng Việt:",
            placeholder="e.g., chất lượng tuyệt vời giao hàng rất nhanh shop tuyệt vời",
            height=120,
            help="Tối đa 2000 ký tự",
            key="single_text"
        )
        char_count = len(user_text)
        st.progress(min(char_count / 2000, 1.0))
        st.caption(f"Ký tự: {char_count}/2000")
    
    with col2:
        st.markdown("### Ví dụ mẫu")
        for label, example in EXAMPLES.items():
            st.button(
                label,
                width="stretch",
                key=f"example_{label}",
                on_click=set_example_text,
                args=(example,)
            )
    
    st.markdown("---")
    
    if user_text:
        suggestions = []
        if "  " in user_text:
            suggestions.append("Xóa khoảng trắng thừa")
        if any(char.isupper() for char in user_text):
            suggestions.append("Chuyển văn bản sang chữ thường để cải thiện kết quả")
        
        if suggestions:
            with st.expander("Gợi ý đầu vào"):
                for suggestion in suggestions:
                    st.info(suggestion)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button("Phân tích Cảm xúc", width="stretch", key="single_analyze")
    
    st.markdown("---")
    
    if analyze_button:
        if not user_text.strip():
            st.error("Vui lòng nhập văn bản để phân tích.")
        else:
            with st.spinner("Đang xử lý phân tích cảm xúc dữ liệu..."):
                result = predict_single(user_text)
                
                if result:
                    sentiment = result.get("sentiment", "Unknown")
                    label = result.get("label", "Unknown")
                    confidence = result.get("confidence", 0)
                    
                    add_to_history(user_text, result)
                    
                    if isinstance(result, dict) and result.get("keywords"):
                        kws_str = ", ".join(result["keywords"])
                    else:
                        # === ĐÃ SỬA CHỮ 'user_input' THÀNH 'user_text' ===
                        words = user_text.split() 
                        keywords_list = [w.strip(".,!?") for w in words if len(w.strip(".,!?")) >= 4]
                        kws_str = ", ".join(keywords_list[:3]) if keywords_list else ", ".join(words[:2])
                    # Style mappings
                    color_map = {"positive": "#d4edda", "neutral": "#fff3cd", "negative": "#f8d7da"}
                    text_color_map = {"positive": "#155724", "neutral": "#856404", "negative": "#721c24"}
                    lbl_lower = label.lower() if label.lower() in ["positive", "neutral", "negative"] else "neutral"
                    
                    # === IN RA HTML CÓ THÊM DÒNG TỪ KHÓA BÊN DƯỚI ===
                    st.markdown(f"""
                    <div style="background-color:{color_map[lbl_lower]}; color:{text_color_map[lbl_lower]}; padding:20px; border-radius:10px; border:1px solid {text_color_map[lbl_lower]}33; margin-bottom:20px;">
                        <h3 style="margin-top:0; color:{text_color_map[lbl_lower]};">Kết quả Phân tích ({MODEL_OPTIONS[st.session_state.selected_model]})</h3>
                        <p style="font-size:18px;"><b>Cảm xúc sắc thái:</b> {sentiment}</p>
                        <p style="font-size:16px;"><b>Keywords:</b> {kws_str}</p>
                        <p style="font-size:16px;"><b>Độ tin cậy thuật toán:</b> {confidence:.2%}</p>
                        
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        copy_text = json.dumps(result, indent=2, ensure_ascii=False)
                        st.download_button("Sao chép kết quả (JSON)", copy_text, file_name=f"sentiment.json", mime="application/json")
                    with col2:
                   
                        csv_data = export_to_csv([{
                            "Text": user_text, "Sentiment": sentiment, "Label": label, "Confidence": f"{confidence:.2%}"
                        }])
                        st.download_button("Xuất file báo cáo CSV", csv_data, file_name=f"sentiment.csv", mime="text/csv")

                    with st.expander("Phản hồi Raw JSON từ API Engine"):
                        st.json(result)
                else:
                    st.error("Phân tích thất bại. Vui lòng kiểm tra lại trạng thái kết nối Terminal API.")

# ══════════════════════════════════════════════════════════════════
# TAB 2: Batch Analysis
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Phân tích hàng loạt dữ liệu")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Lựa chọn 1: Tải lên file CSV nhận xét")
        uploaded_file = st.file_uploader("Chọn file dữ liệu CSV", type="csv", key="batch_csv")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Dữ liệu xem trước cấu trúc:", df.head(3))
            text_column = st.selectbox("Chọn cột chứa câu nhận xét:", df.columns)
            
            if st.button("Kích hoạt Phân tích hàng loạt (CSV)", width="stretch"):
                texts = df[text_column].astype(str).tolist()[:100]
                with st.spinner(f"Đang xử lý phân tích {len(texts)} văn bản..."):
                    batch_result = predict_batch(texts)
                    if batch_result:
                        results = batch_result.get("results", [])
                        latency = batch_result.get("latency_ms", 0)
                        
                        results_df = pd.DataFrame([{
                            "Văn bản gốc": r["text"], "Sắc thái nhận diện": r["sentiment"],
                            "Nhãn nhị phân": r["label"], "Độ tin cậy": f"{r['confidence']:.2%}"
                        } for r in results])
                        
                        for r in results:
                            add_to_history(r["text"], r)
                        
                        st.success(f"Đã phân tích hoàn tất trong {latency}ms")
                        st.dataframe(results_df, width="stretch", hide_index=True)
    
    with col2:
        st.markdown("#### Lựa chọn 2: Dán nhiều dòng nhận xét")
        batch_text = st.text_area(
            "Mỗi dòng tương ứng với một nhận xét riêng biệt (Max 100 dòng):",
            height=200,
            placeholder="Tuyệt vời shop ơi\nko thích hàng này chút nào\ngiao hàng bình thường"
        )
        
        if st.button("Kích hoạt Phân tích hàng loạt (Dán)", width="stretch"):
            texts = [t.strip() for t in batch_text.split("\n") if t.strip()][:100]
            if not texts:
                st.error("Vui lòng cung cấp dữ liệu văn bản để phân tích.")
            else:
                with st.spinner("Đang chạy tính toán ma trận dữ liệu chuỗi..."):
                    batch_result = predict_batch(texts)
                    if batch_result:
                        results = batch_result.get("results", [])
                        
                        # === THÊM HÀM TRÍCH XUẤT TỪ KHÓA ===
                        def get_batch_keywords(text):
                            words = text.split()
                            kws = [w.strip(".,!?") for w in words if len(w.strip(".,!?")) >= 4]
                            return ", ".join(kws[:3]) if kws else ", ".join(words[:2])

                        # === CẬP NHẬT LẠI BẢNG DATAFRAME ===
                        results_df = pd.DataFrame([{
                            "Văn bản gốc": r["text"], 
                            "Sắc thái nhận diện": r["sentiment"],
                            "Nhãn nhị phân": r["label"], 
                            "Độ tin cậy": f"{r['confidence']:.2%}",
                            "Từ khóa": get_batch_keywords(r["text"])  # <--- Khai báo thêm cột mới ở cuối
                        } for r in results])
                        
                        for r in results:
                            add_to_history(r["text"], r)
                        st.success("Xử lý thành công!")
                        st.dataframe(results_df, width="stretch", hide_index=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3: Analytics Graphs
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Thống kê đồ thị trực quan trực tuyến")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Biểu đồ phân bố tỷ lệ Sắc thái cảm xúc")
            sentiment_counts = history_df["label"].value_counts()
            fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                         color_discrete_map={"POSITIVE": "#28a745", "NEUTRAL": "#ffc107", "NEGATIVE": "#dc3545"})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Mật độ phân bổ phổ độ tin cậy")
            fig = px.histogram(history_df, x="confidence", nbins=15, labels={"confidence": "Độ tin cậy"})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Hệ thống chưa ghi nhận lịch sử phân tích trong phiên này để vẽ đồ thị.")
    # ══════════════════════════════════════════════════════════════════
    # Phần Thống kê chung (Đặt dưới biểu đồ của TAB 3 hoặc dưới cùng TAB 4)
    # ══════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown("### Thống kê chung")
    
    # 1. Kéo dữ liệu thực tế từ hệ thống
    history_data = st.session_state.history
    tong_so = len(history_data)
    
    # 2. Làm toán tự động
    if tong_so > 0:
        # Tính độ tin cậy trung bình (key của bạn là 'confidence')
        tong_confidence = sum(item['confidence'] for item in history_data)
        # Nếu confidence đang ở dạng 0.x thì nhân 100, nếu đã là phần trăm thì bỏ * 100 đi nhé
        avg_confidence = (tong_confidence / tong_so) * 100 
        
        # Đếm số lượng nhãn (Sửa chữ 'sentiment' thành 'label' cho khớp với data gốc)
        so_positive = sum(1 for item in history_data if item['label'] == 'POSITIVE')
        so_negative = sum(1 for item in history_data if item['label'] == 'NEGATIVE')
        # Tính tỷ lệ %
        ty_le_pos = (so_positive / tong_so) * 100
        ty_le_neg = (so_negative / tong_so) * 100
    else:
        avg_confidence = 0.0
        ty_le_pos = 0.0
        ty_le_neg = 0.0

    # 3. Đẩy số liệu sống lên giao diện
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Tổng số phân tích", value=f"{tong_so}")
        
    with col2:
        st.metric(label="Độ tin cậy trung bình", value=f"{avg_confidence:.2f}%")
        
    with col3:
        st.metric(label="Tỷ lệ Positive", value=f"{ty_le_pos:.1f}%")
        
    with col4:
        st.metric(label="Tỷ lệ Negative", value=f"{ty_le_neg:.1f}%")

# 
# ══════════════════════════════════════════════════════════════════
# TAB 4: History Table
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Nhật ký lịch sử phân tích hệ thống")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        
        # --- BƯỚC 1: TẠO LOGIC TỰ ĐỘNG TRÍCH XUẤT TỪ KHÓA ---
        def get_keywords(text):
            # Tách câu thành các từ và xóa dấu câu dính liền
            words = text.split()
            # Lọc các từ có độ dài >= 4 ký tự (thường là từ ghép hoặc tính từ quan trọng trong tiếng Việt)
            # Tránh lấy mấy từ rác như "là", "và", "có", "thì"...
            keywords = [w.strip(".,!?") for w in words if len(w.strip(".,!?")) >= 4]
            # Nếu lọc xong mà không có từ nào, lấy đại 2 từ đầu tiên. Nếu có thì lấy tối đa 3 từ khóa.
            return ", ".join(keywords[:3]) if keywords else ", ".join(words[:2])

        # Áp dụng hàm vừa viết để đẻ ra thêm một cột "keyword" vào DataFrame
        history_df["keyword"] = history_df["text"].apply(get_keywords)
        # ----------------------------------------------------

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Xóa sạch nhật ký lịch sử", width="stretch"):
                st.session_state.history = []
                # Nếu bạn có biến trend_data thì giữ, không thì cứ để y cũ
                if 'trend_data' in st.session_state:
                    st.session_state.trend_data = []
                st.rerun()
        
        # --- BƯỚC 2: CẬP NHẬT HIỂN THỊ THÊM CỘT TỪ KHÓA Ở NGOÀI CÙNG ---
        st.dataframe(
            history_df[["timestamp", "text", "sentiment", "confidence", "keyword"]].rename(
                columns={
                    "timestamp": "Thời gian", 
                    "text": "Văn bản gốc", 
                    "sentiment": "Kết quả", 
                    "confidence": "Độ chính xác",
                    "keyword": "Từ khóa"  # <--- Khai báo tên tiếng Việt cho cột mới
                }
            ), 
            width="stretch", 
            hide_index=True
        )
    else:
        st.info("Nhật ký rỗng.")
        

   