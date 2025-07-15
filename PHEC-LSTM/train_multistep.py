import torch
import torch.optim as optim
import torch.nn as nn
from model import MultiStepLSTMModel
from dataset import build_dataset
import logging
import time
import pickle
from plot import plot_loss
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

from evaluation import build_eval_step  # ensure both are imported

def train_lstm_multistep(window_size, batch_size, hidden_size, opt_name='Adam', loss_name='MSE', num_epochs=50,
               multi_step=False, forecast_steps=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("🚀 Khởi tạo mô hình LSTM...")
    logging.info(f"📦 Sử dụng thiết bị: {device}")
    
    # Model output shape tuỳ theo step
    model = MultiStepLSTMModel(input_size=7, hidden_size=hidden_size, num_layers=1, forecast_steps=forecast_steps).to(device)

    optimizers = {'Adam': optim.Adam, 'RMSprop': optim.RMSprop, 'SGD': optim.SGD}
    losses = {'MSE': nn.MSELoss, 'MAE': nn.L1Loss}
    criterion = losses[loss_name]()
    optimizer = optimizers[opt_name](model.parameters(), lr=0.001)
    scaler = GradScaler()

    unique_id = f"{window_size}_{batch_size}_{hidden_size}_{opt_name}_{loss_name}_{num_epochs}_{forecast_steps}"
    model_path = f'model_{unique_id}.pth'

    # Load dataset
    train_loader, val_loader, test_loader = build_dataset(
        unique_id,
        window_size=window_size,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        multi_step=multi_step,
        forecast_steps=forecast_steps
    )

    train_losses_all = []
    val_losses_all = []
    best_val_loss = float('inf')
    patience = 10
    counter = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = model(X_batch)
                if multi_step:
                    loss = criterion(outputs, y_batch)
                else:
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
                    if multi_step:
                        loss = criterion(outputs, y_batch)
                    else:
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

    # Lưu lịch sử loss
    with open(f'loss_histories_{unique_id}.pkl', 'wb') as f:
        pickle.dump({'train_loss': train_losses_all, 'val_loss': val_losses_all}, f)

    # Đánh giá mô hình
    metrics = build_eval_step(model, test_loader, unique_id, forecast_steps)
    plot_loss(unique_id, loss_name)

    result = {
        'window_size': window_size,
        'batch_size': batch_size,
        'hidden_size': hidden_size,
        'optimizer': opt_name,
        'loss': loss_name,
        'epochs': num_epochs,
        'multi_step': multi_step,
        'forecast_steps': forecast_steps,
        **metrics
    }

    with open(f'parameters_{unique_id}.json', 'w') as f:
        json.dump(result, f, indent=4)

    return model_path


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình LSTM")
    parser.add_argument('--config', type=str, nargs=6, help='window_size batch_size hidden_size opt_name loss_name num_epochs')
    return parser.parse_args()



if __name__ == "__main__":

    args = parse_args()
    window_size = int(args.config[0])
    batch_size = int(args.config[1])
    hidden_size = int(args.config[2])
    opt_name = args.config[3]
    loss_name = args.config[4]
    num_epochs = int(args.config[5])

    print(f"📦 Tham số nhận được: batch_size={batch_size}, epochs={num_epochs}, window_size={window_size}")
    train_lstm_multistep(window_size, batch_size, hidden_size, opt_name, loss_name, num_epochs, multi_step=True, forecast_steps=3)