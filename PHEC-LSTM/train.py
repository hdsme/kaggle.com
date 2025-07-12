import torch
import torch.optim as optim
import torch.nn as nn
from model import LSTMModel
from dataset import build_dataset
from evaluation import build_eval
import logging
import os
import time

# AMP cho mixed precision
from torch.amp import autocast, GradScaler

MODEL_PATH = 'ltsm.pth'
LOG_PATH = 'training.log'

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)

def train_lstm():
    logging.info("🚀 Khởi tạo mô hình LSTM...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"📦 Sử dụng thiết bị: {device}")

    model = LSTMModel(input_size=7, hidden_size=50, num_layers=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    patience = 10
    best_val_loss = float('inf')
    counter = 0
    num_epochs = 50

    # Load dataset
    train_loader, val_loader, test_loader = build_dataset(batch_size=128, num_workers=2, pin_memory=True)

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = model(X_batch)
                loss = criterion(outputs.view(-1), y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                with autocast(device_type='cuda'):
                    outputs = model(X_batch)
                    loss = criterion(outputs.view(-1), y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        epoch_time = time.time() - start_time
        logging.info(f'Epoch {epoch+1}/{num_epochs} | ⏱️ {epoch_time:.2f}s | 📉 Train Loss: {train_loss:.6f} | 🧪 Val Loss: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            logging.info(f'✅ Đã lưu mô hình tốt nhất tại epoch {epoch+1} (val_loss={val_loss:.6f})')
        else:
            counter += 1
            logging.info(f'⏸ Không cải thiện. EarlyStopping: {counter}/{patience}')
            if counter >= patience:
                logging.info("⛔ Dừng sớm do không cải thiện.")
                break

    return model


if __name__ == "__main__":
    logging.info("🏁 Bắt đầu huấn luyện LSTM...")
    # model = train_lstm()
    logging.info("✅ Huấn luyện kết thúc.")

    if os.path.exists(MODEL_PATH):
        logging.info("📥 Tải mô hình đã lưu...")
        model = LSTMModel(input_size=7, hidden_size=50, num_layers=1).to(device)
        model.load_state_dict(torch.load(MODEL_PATH))
        logging.info("🧪 Đánh giá mô hình...")
        build_eval(model)
    else:
        logging.error("❌ Không tìm thấy mô hình đã lưu!")
