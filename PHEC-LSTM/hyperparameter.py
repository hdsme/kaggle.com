import torch
import torch.optim as optim
from lstm_model import LSTMModel
import pickle

# Danh sách hyperparameter để thử nghiệm
window_sizes = [30, 60, 120]
hidden_sizes = [32, 50, 100]
batch_sizes = [16, 32, 64]
optimizers = {'Adam': optim.Adam, 'RMSprop': optim.RMSprop, 'SGD': optim.SGD}
losses = {'MSE': nn.MSELoss, 'MAE': nn.L1Loss}
num_epochs_list = [20, 50, 100]

results = []

for window_size in window_sizes:
    # Tạo lại dataset với window_size mới
    dataset = TimeSeriesDataset(scaled_df.values, window_size)
    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.2)
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    for batch_size in batch_sizes:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        for hidden_size in hidden_sizes:
            for opt_name, opt_class in optimizers.items():
                for loss_name, loss_class in losses.items():
                    for num_epochs in num_epochs_list:
                        print(f'\nTesting: window_size={window_size}, batch_size={batch_size}, hidden_size={hidden_size}, optimizer={opt_name}, loss={loss_name}, epochs={num_epochs}')

                        # Khởi tạo mô hình
                        model = LSTMModel(input_size=7, hidden_size=hidden_size, num_layers=1).to(device)
                        criterion = loss_class()
                        optimizer = opt_class(model.parameters(), lr=0.001)

                        # Huấn luyện
                        train_losses = []
                        val_losses = []
                        best_val_loss = float('inf')
                        counter = 0
                        best_model_path = f'best_model_{window_size}_{batch_size}_{hidden_size}_{opt_name}_{loss_name}.pth'

                        for epoch in range(num_epochs):
                            model.train()
                            train_loss = 0
                            for X_batch, y_batch in train_loader:
                                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                                optimizer.zero_grad()
                                outputs = model(X_batch)
                                loss = criterion(outputs.squeeze(), y_batch)
                                loss.backward()
                                optimizer.step()
                                train_loss += loss.item()
                            train_loss /= len(train_loader)
                            train_losses.append(train_loss)

                            model.eval()
                            val_loss = 0
                            with torch.no_grad():
                                for X_batch, y_batch in val_loader:
                                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                                    outputs = model(X_batch)
                                    val_loss += criterion(outputs.squeeze(), y_batch).item()
                                val_loss /= len(val_loader)
                                val_losses.append(val_loss)

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

# In kết quả
for res in results:
    print(f"window_size={res['window_size']}, batch_size={res['batch_size']}, hidden_size={res['hidden_size']}, optimizer={res['optimizer']}, loss={res['loss']}, epochs={res['epochs']}, MAE={res['mae']:.4f}, RMSE={res['rmse']:.4f}, MAPE={res['mape']:.4f}%")


def build_hyperparams():
    # Tải kết quả
    with open('hyperparameter_results.pkl', 'rb') as f:
        results = pickle.load(f)

    # Chuyển sang DataFrame
    df_results = pd.DataFrame(results)

    # Tìm bộ tham số có MAE thấp nhất
    best_result = df_results.loc[df_results['mae'].idxmin()]

    print("Best parameters:")
    print(f"window_size: {best_result['window_size']}")
    print(f"batch_size: {best_result['batch_size']}")
    print(f"hidden_size: {best_result['hidden_size']}")
    print(f"optimizer: {best_result['optimizer']}")
    print(f"loss: {best_result['loss']}")
    print(f"epochs: {best_result['epochs']}")
    print(f"MAE: {best_result['mae']:.4f}")
    print(f"RMSE: {best_result['rmse']:.4f}")
    print(f"MAPE: {best_result['mape']:.4f}%")

    # Lưu kết quả
    df_results.to_csv('hyperparameter_results.csv', index=False)