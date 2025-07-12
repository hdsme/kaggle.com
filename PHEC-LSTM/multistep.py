import torch
import torch.nn as nn
from lstm_model import LSTMModel

# Sửa đổi dataset để dự đoán nhiều bước
class MultiStepDataset(Dataset):
    def __init__(self, data, window_size, forecast_steps=3):
        self.data = data
        self.window_size = window_size
        self.forecast_steps = forecast_steps

    def __len__(self):
        return len(self.data) - self.window_size - self.forecast_steps + 1

    def __getitem__(self, idx):
        X = self.data[idx:idx + self.window_size]
        y = self.data[idx + self.window_size:idx + self.window_size + self.forecast_steps, 0]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Tạo dataset
forecast_steps = 3
multi_dataset = MultiStepDataset(scaled_df.values, window_size=60, forecast_steps=forecast_steps)
train_size = int(len(multi_dataset) * 0.7)
val_size = int(len(multi_dataset) * 0.2)
test_size = len(multi_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(multi_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Sửa đổi mô hình để dự đoán nhiều bước
class MultiStepLSTMModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=50, num_layers=1, forecast_steps=3):
        super(MultiStepLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, forecast_steps)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Huấn luyện mô hình
model = MultiStepLSTMModel(input_size=7, hidden_size=50, num_layers=1, forecast_steps=forecast_steps).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
val_losses = []
patience = 10
best_val_loss = float('inf')
counter = 0
best_model_path = 'best_multi_step_model.pth'

for epoch in range(50):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
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
            val_loss += criterion(outputs, y_batch).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

# Đánh giá
model.load_state_dict(torch.load(best_model_path))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        y_pred.extend(outputs.cpu().numpy())
        y_true.extend(y_batch.cpu().numpy())

# Đảo ngược chuẩn hóa và tính MAE cho từng bước
y_pred = np.array(y_pred)
y_true = np.array(y_true)
for step in range(forecast_steps):
    y_pred_step = y_pred[:, step].reshape(-1, 1)
    y_true_step = y_true[:, step].reshape(-1, 1)
    y_pred_inv = scaler.inverse_transform(np.concatenate((y_pred_step, np.zeros((y_pred_step.shape[0], 6))), axis=1))[:, 0]
    y_true_inv = scaler.inverse_transform(np.concatenate((y_true_step, np.zeros((y_true_step.shape[0], 6))), axis=1))[:, 0]
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    print(f'MAE for step {step+1}: {mae:.4f}')
