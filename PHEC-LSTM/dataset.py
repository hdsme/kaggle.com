import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import requests
import zipfile
import io, os


# Tạo dataset PyTorch
class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        X = self.data[idx:idx + self.window_size]
        y = self.data[idx + self.window_size, 0]  # Dự đoán Global_active_power
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def download_dataset():
    url = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
    response = requests.get(url)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall("datasets")
    print("Download and extraction completed.")
    return os.path.abspath('datasets')

def build_dataset(window_size=60, batch_size=32, flag=0):
    """
    flag = 0 -> singlestep
    flag = 1 -> multistep
    """
    dataset = download_dataset()
    # Đọc dữ liệu
    data = pd.read_csv(f'{dataset}/household_power_consumption.txt', sep=';', parse_dates={'datetime': ['Date', 'Time']}, infer_datetime_format=True, low_memory=False)
    data = data[['datetime', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]

    # Xử lý giá trị thiếu
    data = data.replace('?', np.nan)
    data = data.fillna(data.mean(numeric_only=True))

    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.iloc[:, 1:])
    features = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    scaled_df = pd.DataFrame(scaled_data, columns=features)

    # Tạo dữ liệu
    window_size = 60  # Thử nghiệm với 30, 60, 120
    dataset = TimeSeriesDataset(scaled_df.values, window_size)

    # Chia dữ liệu
    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.2)
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Tạo DataLoader
    batch_size = 32  # Thử nghiệm với 16, 32, 64
    train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train, val, test
