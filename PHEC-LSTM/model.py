import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=50, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)  # out: (batch, window_size, hidden_size)
        out = self.dropout(out[:, -1, :])  # Lấy output của bước cuối
        out = self.fc(out)  # (batch, 1)
        return out