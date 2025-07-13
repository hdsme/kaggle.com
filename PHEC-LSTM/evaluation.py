import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import torch
import joblib
import json
def build_eval(model, test_loader, unique_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = joblib.load(f'scaler_{unique_id}.save')
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
    metrics = {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4)
    }
    with open(f'metrics_{unique_id}.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    # Vẽ so sánh dự đoán và thực tế
    plt.figure(figsize=(15, 5))
    plt.plot(y_true_inv[:200], label='Actual')
    plt.plot(y_pred_inv[:200], label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.savefig(f'prediction_plot_{unique_id}.png')
    plt.show()

    return metrics