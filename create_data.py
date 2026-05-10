import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ------------------ 1. 设备配置 ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------ 2. CNN 模型定义 (与之前相同) ------------------
class WeatherCNN(nn.Module):
    def __init__(self, num_features=4, seq_length=12):
        super().__init__()
        self.conv1 = nn.Conv1d(num_features, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        flatten_dim = 16 * (seq_length // 2)
        self.fc1 = nn.Linear(flatten_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ------------------ 3. 纯数组 FFT 对齐 + 预处理 ------------------
def fft_filter_and_standardize(data, keep_low_freq_ratio=0.1):
    """
    data: (time_steps, 4) 的 numpy 数组，四个变量
    对每个变量单独做 FFT 低通滤波，然后全表标准化
    返回处理后的数组 (time_steps, 4) 和 scaler 参数（均值、标准差）
    """
    filtered = np.zeros_like(data)
    for ch in range(data.shape[1]):
        signal = data[:, ch]
        n = len(signal)
        fft_vals = np.fft.fft(signal)
        freqs = np.fft.fftfreq(n)
        # 只保留低频：频率绝对值小于最高频率 * keep_low_freq_ratio 的成分
        cutoff = keep_low_freq_ratio * np.max(np.abs(freqs))
        fft_vals[np.abs(freqs) > cutoff] = 0
        filtered[:, ch] = np.real(np.fft.ifft(fft_vals))

    # Z-score 标准化
    mean = filtered.mean(axis=0, keepdims=True)
    std = filtered.std(axis=0, keepdims=True)
    scaled = (filtered - mean) / std
    return scaled, mean.flatten(), std.flatten()

# ------------------ 4. 主程序：生成数组数据并测试 ------------------
if __name__ == "__main__":
    # ----- 4.1 生成模拟气象数据 (100个时间步，间隔5分钟) -----
    time_steps = 200000
    t = np.arange(time_steps)

    # 构造带有日周期（每288步约24小时）和噪声的数据
    pressure = 101325 + 50 * np.sin(2*np.pi * t / 288) + np.random.normal(0, 20, time_steps)
    wind_speed = 3.0 + 1.5 * np.sin(2*np.pi * t / 144) + np.random.normal(0, 0.8, time_steps)
    temperature = 20 + 5 * np.sin(2*np.pi * t / 288) + np.random.normal(0, 2, time_steps)
    humidity = 60 + 10 * np.sin(2*np.pi * t / 200) + np.random.normal(0, 5, time_steps)

    raw_data = np.column_stack((pressure, wind_speed, temperature, humidity))
    print("原始数据形状:", raw_data.shape)   # (100, 4)

    # ----- 4.2 FFT 对齐 + 滤波 + 标准化 -----
    processed_data, mean_vals, std_vals = fft_filter_and_standardize(raw_data, keep_low_freq_ratio=0.15)
    print("处理后数据形状:", processed_data.shape)
    print("均值:", mean_vals, "标准差:", std_vals)

    # 生成 200 组处理后的数据并保存到 data.txt 中
    np.savetxt('./data.txt', processed_data, fmt='%.6f', delimiter=',',
               header='pressure,wind_speed,temperature,humidity', comments='')
    print(f"处理后的 {len(processed_data)} 组数据已成功保存至 ./data.txt")

    # ----- 4.3 构建滑动窗口序列 -----
    seq_length = 12   # 用过去12个时间步预测下一时刻温度
    X, Y = [], []
    for i in range(len(processed_data) - seq_length):
        X.append(processed_data[i:i+seq_length])          # [12,4]
        Y.append(processed_data[i+seq_length, 2])         # 温度是第3列 (索引2)

    X = np.array(X)  # (样本数, 12, 4)
    Y = np.array(Y)  # (样本数,)
    print(f"X 形状: {X.shape}, Y 形状: {Y.shape}")
