import torch
import torch.optim as optim
from model import LSTMModel
from dataset import build_dataset()


MODEL_PATH = 'ltsm.pth'

def train_ltsm():
    model = LSTMModel(input_size=7, hidden_size=50, num_layers=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()  # Thử nghiệm với nn.L1Loss() cho MAE
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Thử nghiệm với RMSprop, SGD
    # Early Stopping
    patience = 10
    best_val_loss = float('inf')
    counter = 0

    # Huấn luyện
    num_epochs = 50  # Thử nghiệm với 20, 50, 100
    train, val, test = build_dataset()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:  # Giả sử train_loader đã được định nghĩa
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:  # Giả sử val_loader đã được định nghĩa
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs.squeeze(), y_batch).item()
            val_loss /= len(val_loader)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

# Tải mô hình tốt nhất
model.load_state_dict(torch.load(MODEL_PATH))