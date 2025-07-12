import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import os, zipfile, requests, io


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        X = self.data[idx:idx + self.window_size]
        y = self.data[idx + self.window_size, 0]  # chỉ dự đoán Global_active_power
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def download_dataset():
    url = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall("datasets")
    return os.path.abspath("datasets")


def build_dataset(window_size=60, batch_size=64):
    dataset_path = download_dataset()

    print("🔁 Đọc dữ liệu gốc...")
    df = pd.read_csv(
        f"{dataset_path}/household_power_consumption.txt",
        sep=';',
        low_memory=False,
        na_values='?',
        dtype={'Global_active_power': 'float32',
               'Global_reactive_power': 'float32',
               'Voltage': 'float32',
               'Global_intensity': 'float32',
               'Sub_metering_1': 'float32',
               'Sub_metering_2': 'float32',
               'Sub_metering_3': 'float32'}
    )

    # Kết hợp Date và Time
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.drop(['Date', 'Time'], axis=1, inplace=True)

    # Sắp xếp theo thời gian và drop NaN
    df.sort_values('datetime', inplace=True)
    df.dropna(inplace=True)

    # Đặt datetime làm index nếu cần (tuỳ yêu cầu mô hình)
    df.reset_index(drop=True, inplace=True)

    print("✅ Dữ liệu sau khi xử lý:", df.shape)

    # Chuẩn hóa chỉ các cột số (bỏ datetime)
    features = df.drop(columns=['datetime']).values
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # Tạo Dataset
    full_dataset = TimeSeriesDataset(scaled_features, window_size)

    # Chia tập train / val / test
    total_len = len(full_dataset)
    train_size = int(total_len * 0.7)
    val_size = int(total_len * 0.2)
    test_size = total_len - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Tạo DataLoader (tối ưu)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader
