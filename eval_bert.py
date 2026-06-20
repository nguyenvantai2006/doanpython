import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

# Khai báo thư mục src để tẹo nữa xài code cũ của nhóm
sys.path.append('./src')

print("🚀 Bắt đầu chiến dịch khảo thí điểm số BERT...")

# 1. Chuẩn bị đề thi (Lấy 20% data để test, giống tỷ lệ train thông thường)
try:
    df = pd.read_csv("data/cleaned_data.csv")
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"✅ Đã lấy {len(test_df)} câu bình luận để làm đề thi!")
except Exception as e:
    print("❌ Lỗi không tìm thấy file data:", e)
    exit()

# 2. Đánh thức bộ não BERT
print("⏳ Đang kéo BERT từ ổ cứng lên RAM...")
try:
    model_path = "models/bert_sentiment/demo/bert_model.pt"
    # Ép nó nạp bằng CPU vì máy không có GPU chuyên dụng
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval() # Chuyển sang chế độ đi thi (không học thêm nữa)
    
    print("✅ Nạp thành công bộ não BERT!")
    print("\n🎉 Tuyệt vời! Tới bước này, mô hình đã sẵn sàng.")
    print("👉 Bây giờ Tài chỉ cần mở file 'src/train_bert.py', copy cái đoạn code vòng lặp dự đoán (predict/evaluate) dán tiếp vào đây là lụm điểm!")
    
except Exception as e:
    print("\n❌ Nạp model thất bại! Lỗi:", type(e).__name__)
    print("\n🔍 BẮT BỆNH: Máy tính khóc thét vì file .pt này chỉ chứa 'trọng số' (state_dict), chứ nó không nhớ bản thiết kế của mạng nơ-ron.")
    print("🛠️ CÁCH SỬA: Bạn cần xem trong thư mục 'src' file nào chứa kiến trúc BERT, rồi import nó lên đầu file này (Ví dụ: from src.bert_model import MyBertClass) là máy sẽ đọc được ngay!")