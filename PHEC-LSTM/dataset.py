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
    dataset_path = download_dataset()

    # Đọc dữ liệu
    data = pd.read_csv(
        f'{dataset_path}/household_power_consumption.txt',
        sep=';',
        parse_dates={'datetime': ['Date', 'Time']},
        low_memory=False,
        dayfirst=True  # ⚠️ Quan trọng vì dữ liệu dạng dd/mm/yyyy
    )

    # Lọc cột cần thiết
    data = data[['datetime', 'Global_active_power', 'Global_reactive_power', 'Voltage',
                 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]

    # Xử lý thiếu
    # Thay thế '?' thành NaN rõ ràng
    data.replace('?', np.nan, inplace=True)

    # Ép từng cột sang float, báo lỗi nếu thất bại
    for col in data.columns[1:]:
        try:
            data[col] = data[col].astype(float)
        except ValueError:
            print(f"⚠️ Lỗi khi chuyển cột {col} sang float")
            print(data[col].unique())

    # In số lượng NaN trước khi fill
    print("NaN trước khi fill:\n", data.isna().sum())

    # Điền giá trị thiếu
    data.fillna(data.mean(numeric_only=True), inplace=True)

    # Kiểm tra lại
    print("NaN sau khi fill:\n", data.isna().sum())
    assert not data.isnull().values.any(), "❌ Vẫn còn NaN trong dữ liệu!"

    # Kiểm tra lại
    assert not data.isnull().values.any(), "❌ Vẫn còn NaN trong dữ liệu!"
    print("✅ Dữ liệu sạch. Không còn NaN.")

    # Chuẩn hóa dữ liệu (bỏ datetime)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.iloc[:, 1:])
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns[1:])

    # Tạo dataset
    dataset = TimeSeriesDataset(scaled_df.values, window_size)

    # Chia tập train/val/test
    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.2)
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Tạo DataLoader
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader

