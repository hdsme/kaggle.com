import torch
import torch.optim as optim
import torch.nn as nn
from model import LSTMModel
from dataset import build_dataset
from evaluation import build_eval
import logging
import os
import time
import pickle
from plot import plot_loss
# AMP cho mixed precision
from torch.amp import autocast, GradScaler
import json
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

def train_lstm(window_size, batch_size, hidden_size, opt_name='Adam', loss_name = 'MSE', num_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("🚀 Khởi tạo mô hình LSTM...")
    logging.info(f"📦 Sử dụng thiết bị: {device}")
    model = LSTMModel(input_size=7, hidden_size=hidden_size, num_layers=1).to(device)
    optimizers = {'Adam': optim.Adam, 'RMSprop': optim.RMSprop, 'SGD': optim.SGD}
    losses = {'MSE': nn.MSELoss, 'MAE': nn.L1Loss}
    criterion = losses[loss_name]()
    optimizer = optimizers[opt_name](model.parameters(), lr=0.001)
    scaler = GradScaler()
    unique_id = f"{window_size}_{batch_size}_{hidden_size}_{opt_name}_{loss_name}"
    model_path = f'model_{unique_id}.pth'
    patience = 10
    best_val_loss = float('inf')
    counter = 0

    # Load dataset
    train_loader, val_loader, test_loader = build_dataset(unique_id, window_size=window_size, batch_size=batch_size, num_workers=2, pin_memory=True)
    train_losses_all = []
    val_losses_all = []
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

        train_losses_all.append(train_loss)
        val_losses_all.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), model_path)
            logging.info(f'✅ Đã lưu mô hình tốt nhất tại epoch {epoch+1} (val_loss={val_loss:.6f})')
        else:
            counter += 1
            logging.info(f'⏸ Không cải thiện. EarlyStopping: {counter}/{patience}')
            if counter >= patience:
                logging.info("⛔ Dừng sớm do không cải thiện.")
                break
    
    with open(f'loss_histories_{unique_id}.pkl', 'wb') as f:
        pickle.dump({'train_loss': train_losses_all, 'val_loss': val_losses_all}, f)
    metrics = build_eval(model, test_loader,unique_id)
    plot_loss(unique_id, loss_name)

    result = {
        'window_size': window_size,
        'batch_size': batch_size,
        'hidden_size': hidden_size,
        'optimizer': opt_name,
        'loss': loss_name,
        'epochs': num_epochs,
        **metrics
    }
    with open(f'parameters_{unique_id}.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    return model_path

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình LSTM")

    parser.add_argument('--eval_only', type=bool, default=False, help='Train or Eval')
    parser.add_argument('--tune', type=bool, default=False, help='Train or Eval')
    parser.add_argument('--hidden_size', type=int, default=50, help='Hidden size cho Model')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size cho DataLoader')
    parser.add_argument('--epochs', type=int, default=50, help='Số lượng epochs huấn luyện')
    parser.add_argument('--window_size', type=int, default=60, help='Kích thước cửa sổ chuỗi thời gian')

    return parser.parse_args()
    

if __name__ == "__main__":
    window_sizes = [30, 60, 120]
    hidden_sizes = [32, 50, 100]
    batch_sizes = [16, 32, 64]
    optimizers = {'Adam': optim.Adam, 'RMSprop': optim.RMSprop, 'SGD': optim.SGD}
    losses = {'MSE': nn.MSELoss, 'MAE': nn.L1Loss}
    num_epochs_list = [1, 10, 20, 50]

    for window_size in window_sizes:
        print(f"\n🌐 Đang xử lý window_size={window_size}")
        for batch_size in batch_sizes:
            print(f"\n🌐 Đang xử lý batch_size={batch_size}")
            for hidden_size in hidden_sizes:
                print(f"\n🌐 Đang xử lý hidden_size={hidden_size}")
                for opt_name, opt_class in optimizers.items():
                    print(f"\n🌐 Đang xử lý opt_name={opt_name}")
                    for loss_name, loss_class in losses.items():
                        print(f"\n🌐 Đang xử lý loss_name={loss_name}")
                        for num_epochs in num_epochs_list:
                            train_lstm(window_size, batch_size, hidden_size, opt_name, loss_name, num_epochs)
    # args = parse_args()
    # print(f"📦 Tham số nhận được: batch_size={args.batch_size}, epochs={args.epochs}, window_size={args.window_size}")

    # if not args.tune:
    #     logging.info("🏁 Bắt đầu huấn luyện LSTM...")
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     model = LSTMModel(input_size=7, hidden_size=args.hidden_size, num_layers=1).to(device)
    #     if not args.eval_only:
    #         train_lstm(window_size=args.window_size, batch_size=args.batch_size, num_epochs=args.epochs)
    #     logging.info("✅ Huấn luyện kết thúc.")

    #     if os.path.exists(MODEL_PATH):
    #         logging.info("📥 Tải mô hình đã lưu...")
    #         model.load_state_dict(torch.load(MODEL_PATH))
    #         logging.info("🧪 Đánh giá mô hình...")
    #         build_eval(model, device, dataset[2])
    #     else:
    #         logging.error("❌ Không tìm thấy mô hình đã lưu!")
    # else:
    #     logging.info("🏁 Bắt đầu tune LSTM...")
