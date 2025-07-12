# [Homework#03] Dự Đoán Mức Tiêu Thụ Điện Năng dùng LSTM
# Học Viên: Hoàng Hào
> **Hướng** **Dẫn:** **Dự** **Đoán** **Mức** **Tiêu** **Thụ** **Điện**
> **Năng** **Hộ** **Gia** **Đình** **Bằng** **LSTM**

**1.** **Mục** **Tiêu**

> Sử dụng dữ liệu điện năng của một hộ gia đình để dự đoán mức tiêu thụ
> điện năng trong tương lai gần.
>
> Ứng dụng mô hình LSTM cho bài toán dự báo chuỗi thời gian.
>
> So sánh hiệu năng khi thay đổi các hyperparameter như window_size,
> units, batch_size,...

**2.** **Kiến** **Thức** **Cần** **Có**

**A.** **Về** **Dữ** **Liệu** **Thời** **Gian**

> Hiểu về chuỗi thời gian (time series).
>
> Khái niệm trễ thời gian (lag), sliding window.

**B.** **Về** **Deep** **Learning**

> Kiến thức cơ bản về Mạng Nơ-ron Nhân Tạo (NN).
>
> Cấu trúc của RNN và lý do dùng LSTM để xử lý chuỗi dài.
>
> Hiểu cách hoạt động của các gate trong LSTM: forget, input, output
> gate.

**C.** **Về** **Kỹ** **Thuật**

> Biết sử dụng thư viện Python: pandas, numpy, matplotlib, scikit-learn,
> tensorflow.keras.
>
> Biết cách chuẩn hóa dữ liệu, chia train/test, đánh giá mô hình bằng
> MAE.

**3.** **Dữ** **Liệu**

Nguồn:

[<u>https://archive.ics.uci.edu/static/public/235/individual+househ</u>](https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip)
[<u>old+electric+power+consumption.zip</u>](https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip)

**Cột** **sử** **dụng:**

\['Global_active_power', 'Global_reactive_power', 'Voltage',
'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
'Sub_metering_3'\]

**4.** **Xây** **Dựng** **Mô** **Hình** **LSTM**

Tạo mô hình tuần tự gồm 2 lớp:

> **LSTM**: dùng khoảng 50 đơn vị, nhận đầu vào có window_size bước và 7
> đặc trưng.
>
> **Dense**: 1 node để dự đoán Global_active_power.

Biên dịch mô hình với:

> Tối ưu: adam
>
> Hàm mất mát: mse

Yêu cầu: Thay đổi các tham số (epochs, window_size, batch_size, và
units), hàm tối ưu, hàm mất mát và rút ra nhận xét sự ảnh hưởng đến độ
chính xác của mô hình

**5.** **Tài** **liệu** **tham** **khảo:**

**<u>LSTM</u>**

<img src="./static/hcpgg0to.png"
style="width:0.19444in;height:0.19444in" />👉
[<u>https://colah.github.io/posts/2015-08-Understanding-LSTMs/</u>](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

**<u>CS231n - Stanford (chương 1 và 2):</u>**

<img src="./static/0thdjm2w.png"
style="width:0.19444in;height:0.19444in" />👉
[<u>https://cs231n.github.io/neural-networks-1/</u>](https://cs231n.github.io/neural-networks-1/)

<img src="./static/rrqbniqd.png"
style="width:0.19444in;height:0.19444in" />👉
[<u>https://cs231n.github.io/neural-networks-2/</u>](https://cs231n.github.io/neural-networks-2/)

(Giải thích chi tiết về forward pass, loss function, backpropagation)
