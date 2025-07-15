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


def build_eval_step(model, test_loader, unique_id, forecast_steps):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = joblib.load(f'scaler_{unique_id}.save')

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)  # (batch_size, forecast_steps)
            y_pred.append(outputs.cpu().numpy())
            y_true.append(y_batch.cpu().numpy())

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    metrics = {}

    for step in range(forecast_steps):
        y_p = y_pred[:, step].reshape(-1, 1)
        y_t = y_true[:, step].reshape(-1, 1)

        # Inverse scale
        pad = np.zeros((y_p.shape[0], scaler.scale_.shape[0] - 1))
        y_p_inv = scaler.inverse_transform(np.concatenate((y_p, pad), axis=1))[:, 0]
        y_t_inv = scaler.inverse_transform(np.concatenate((y_t, pad), axis=1))[:, 0]

        # Tính chỉ số
        mae = mean_absolute_error(y_t_inv, y_p_inv)
        rmse = np.sqrt(mean_squared_error(y_t_inv, y_p_inv))
        mape = np.mean(np.abs((y_t_inv - y_p_inv) / (y_t_inv + 1e-10))) * 100

        print(f'Step {step+1} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%')

        metrics[f"Step_{step+1}"] = {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "MAPE": round(mape, 2)
        }

        # Vẽ biểu đồ cho từng step
        plt.figure(figsize=(15, 5))
        plt.plot(y_t_inv[:200], label='Actual')
        plt.plot(y_p_inv[:200], label=f'Predicted (Step {step+1})')
        plt.xlabel('Time')
        plt.ylabel('Power (kW)')
        plt.title(f'Actual vs Predicted - Step {step+1}')
        plt.legend()
        plt.savefig(f'prediction_plot_{unique_id}_step_{step+1}.png')
        plt.close()

    # Lưu các metrics
    with open(f'metrics_{unique_id}.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics
