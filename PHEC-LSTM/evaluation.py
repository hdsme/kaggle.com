import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
from dataset import build_dataset

def build_eval(model, device):
    scaler = MinMaxScaler()
    train_loader, val_loader, test_loader = build_dataset(batch_size=128, num_workers=2, pin_memory=True)
    # Đánh giá mô hình
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze()
            y_pred.extend(outputs.cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())

    # Đảo ngược chuẩn hóa
    y_pred = np.array(y_pred).reshape(-1, 1)
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred_inv = scaler.inverse_transform(np.concatenate((y_pred, np.zeros((y_pred.shape[0], 6))), axis=1))[:, 0]
    y_true_inv = scaler.inverse_transform(np.concatenate((y_true, np.zeros((y_true.shape[0], 6))), axis=1))[:, 0]

    # Tính các chỉ số
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    mape = np.mean(np.abs((y_true_inv - y_pred_inv) / (y_true_inv + 1e-10))) * 100  # Thêm 1e-10 để tránh chia cho 0

    print(f'Test MAE: {mae:.4f}')
    print(f'Test RMSE: {rmse:.4f}')
    print(f'Test MAPE: {mape:.4f}%')

    # Vẽ so sánh dự đoán và thực tế
    plt.figure(figsize=(15, 5))
    plt.plot(y_true_inv[:200], label='Actual')
    plt.plot(y_pred_inv[:200], label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Global_active_power (kW)')
    plt.title('Actual vs Predicted Global_active_power')
    plt.legend()
    plt.savefig('prediction_plot.png')
    plt.show()