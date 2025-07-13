import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import pickle
from model import LSTMModel
from dataset import build_dataset

import joblib
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Danh sách hyperparameter để thử nghiệm
window_sizes = [30, 60, 120]
hidden_sizes = [32, 50, 100]
batch_sizes = [16, 32, 64]
optimizers = {'Adam': optim.Adam, 'RMSprop': optim.RMSprop, 'SGD': optim.SGD}
losses = {'MSE': nn.MSELoss, 'MAE': nn.L1Loss}
num_epochs_list = [1, 20, 50, 100]

results = []

for window_size in window_sizes:
    print(f"\n🌐 Đang xử lý window_size={window_size}")
    for batch_size in batch_sizes:
        print(f"\n🌐 Đang xử lý batch_size={batch_size}")
        for hidden_size in hidden_sizes:
            print(f"\n🌐 Đang xử lý hidden_size={hidden_size}")
            train_loader, val_loader, test_loader = build_dataset(window_size=window_size, batch_size=batch_size, num_workers=2, pin_memory=True)
            for opt_name, opt_class in optimizers.items():
                print(f"\n🌐 Đang xử lý opt_name={opt_name}")
                for loss_name, loss_class in losses.items():
                    print(f"\n🌐 Đang xử lý loss_name={loss_name}")
                    for num_epochs in num_epochs_list:
                        print(f'🧪 Testing: ws={window_size}, bs={batch_size}, hs={hidden_size}, opt={opt_name}, loss={loss_name}, epochs={num_epochs}')

                        model = LSTMModel(input_size=7, hidden_size=hidden_size, num_layers=1).to(device)
                        criterion = loss_class()
                        optimizer = opt_class(model.parameters(), lr=0.001)

                        best_val_loss = float('inf')
                        counter = 0
                        train_losses, val_losses = [], []
                        best_model_path = f'model_{window_size}_{batch_size}_{hidden_size}_{opt_name}_{loss_name}.pth'

                        for epoch in range(num_epochs):
                            model.train()
                            train_loss = 0.0
                            for X_batch, y_batch in train_loader:
                                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                                optimizer.zero_grad()
                                outputs = model(X_batch)
                                loss = criterion(outputs.view(-1), y_batch)
                                loss.backward()
                                optimizer.step()
                                train_loss += loss.item()
                            train_loss /= len(train_loader)
                            train_losses.append(train_loss)

                            # Validation
                            model.eval()
                            val_loss = 0.0
                            with torch.no_grad():
                                for X_batch, y_batch in val_loader:
                                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                                    outputs = model(X_batch)
                                    val_loss += criterion(outputs.view(-1), y_batch).item()
                            val_loss /= len(val_loader)
                            val_losses.append(val_loss)

                            # Early stopping
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                counter = 0
                                torch.save(model.state_dict(), best_model_path)
                            else:
                                counter += 1
                                if counter >= 10:
                                    break

                        # Đánh giá trên tập test
                        model.load_state_dict(torch.load(best_model_path))
                        model.eval()
                        y_true, y_pred = [], []

                        with torch.no_grad():
                            for X_batch, y_batch in test_loader:
                                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                                outputs = model(X_batch).squeeze()
                                y_pred.extend(outputs.cpu().numpy())
                                y_true.extend(y_batch.cpu().numpy())

                        y_pred = np.array(y_pred).reshape(-1, 1)
                        y_true = np.array(y_true).reshape(-1, 1)

                        
                        scaler = joblib.load(f'scaler_{window_size}_{batch_size}.save')

                        # Inverse scale
                        y_pred_inv = scaler.inverse_transform(np.concatenate((y_pred, np.zeros((y_pred.shape[0], 6))), axis=1))[:, 0]
                        y_true_inv = scaler.inverse_transform(np.concatenate((y_true, np.zeros((y_true.shape[0], 6))), axis=1))[:, 0]

                        mae = mean_absolute_error(y_true_inv, y_pred_inv)
                        rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
                        mape = np.mean(np.abs((y_true_inv - y_pred_inv) / (y_true_inv + 1e-10))) * 100

                        results.append({
                            'window_size': window_size,
                            'batch_size': batch_size,
                            'hidden_size': hidden_size,
                            'optimizer': opt_name,
                            'loss': loss_name,
                            'epochs': num_epochs,
                            'mae': mae,
                            'rmse': rmse,
                            'mape': mape
                        })

# Lưu kết quả
with open('hyperparameter_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# In ra kết quả
for res in results:
    print(f"✔️ ws={res['window_size']}, bs={res['batch_size']}, hs={res['hidden_size']}, "
          f"opt={res['optimizer']}, loss={res['loss']}, epochs={res['epochs']}, "
          f"MAE={res['mae']:.4f}, RMSE={res['rmse']:.4f}, MAPE={res['mape']:.4f}%")
