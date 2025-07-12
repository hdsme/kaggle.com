import torch
import torch.optim as optim
from model import LSTMModel
from dataset import build_dataset
from evaluation import build_eval
import torch.nn as nn
import logging
import os

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

def train_ltsm():
    logging.info("Khởi tạo mô hình LSTM...")
    model = LSTMModel(input_size=7, hidden_size=50, num_layers=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    patience = 10
    best_val_loss = float('inf')
    counter = 0
    num_epochs = 50

    logging.info(f"Thông số huấn luyện: epochs={num_epochs}, patience={patience}, optimizer=Adam, loss=MSELoss")
    logging.info(f"Sử dụng thiết bị: {device}")

    train, val, test = build_dataset()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs.squeeze(), y_batch).item()
            val_loss /= len(val)

        logging.info(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            logging.info(f'Đã lưu mô hình tốt nhất tại epoch {epoch+1} với val_loss={val_loss:.6f}')
        else:
            counter += 1
            logging.info(f'Không cải thiện. EarlyStopping Counter: {counter}/{patience}')
            if counter >= patience:
                logging.info("Dừng sớm do không cải thiện validation loss.")
                break

    return model

if __name__ == "__main__":
    logging.info("Bắt đầu huấn luyện LSTM...")
    model = train_ltsm()
    logging.info("Huấn luyện kết thúc.")

    if os.path.exists(MODEL_PATH):
        logging.info("Tải mô hình LSTM đã lưu...")
        model.load_state_dict(torch.load(MODEL_PATH))
        logging.info("Đánh giá mô hình LSTM...")
        build_eval(model)
    else:
        logging.error("Không tìm thấy mô hình đã lưu!")
