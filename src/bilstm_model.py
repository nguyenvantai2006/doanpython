import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, output_dim=3, n_layers=2):
        super(BiLSTMClassifier, self).__init__()
        # Tầng nhúng từ (Embedding) biến các ID từ thành vector số
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Tầng BiLSTM chạy 2 chiều xử lý chuỗi ngữ cảnh câu
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=n_layers, 
                            bidirectional=True,  # Chạy 2 chiều ngược xuôi
                            batch_first=True,
                            dropout=0.3)
        
        # Tầng kết nối đầy đủ (Fully Connected) để phân loại sang 3 nhãn (POS, NEU, NEG)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        
        # Lấy trạng thái ẩn cuối cùng của cả 2 chiều gom lại làm một
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)