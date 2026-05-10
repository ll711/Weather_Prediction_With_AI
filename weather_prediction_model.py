import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.signal import windows
import matplotlib.pyplot as plt

# 1. Check and configure NVIDIA GPU acceleration (CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} (NVIDIA GPU Acceleration: {torch.cuda.is_available()})")


# 2. Define 1D CNN model to process meteorological time series features
class WeatherCNN(nn.Module):
    def __init__(self, num_features=4, seq_length=12):
        super(WeatherCNN, self).__init__()
        # 1D-CNN Architecture: Deeper layers with MaxPooling and BatchNorm
        # Block 1
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Block 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # Calculate flattened dimension after 2 pooling layers
        # sequence length is halved *twice* by the two pool layers
        # seq_length=12 -> pool1(6) -> pool2(3)
        flatten_dim = 32 * (seq_length // 4)
        self.fc1 = nn.Linear(flatten_dim, 64)
        self.fc2 = nn.Linear(64, 1)  # Predicting one future value

    def forward(self, x):
        # x shape: (batch_size, num_features, seq_length)
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        # Classifier
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def preprocess_and_fft_align(timestamps, features_data, interval='5T'):
    """
    1. pandas 5-minute resampling alignment
    2. numpy.fft frequency domain analysis/denoising
    3. sklearn standardization
    """
    df = pd.DataFrame(features_data, index=pd.to_datetime(timestamps),
                      columns=['pressure', 'wind_speed', 'temperature', 'humidity'])

    # 1. Align data: 5-minute resampling and interpolation
    df = df.resample(interval).mean().interpolate(method='linear')
    df = df.fillna(method='bfill').fillna(method='ffill')

    # 2. FFT denoising and filtering
    filtered_data = {}
    for col in df.columns:
        signal = df[col].values
        n = len(signal)
        # Execute FFT
        freq_domain = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(n)

        # Zero out high-frequency parts (simple low-pass filtering)
        # Assume keeping top 10% frequency components, the rest as noise
        threshold = 0.1 * np.max(np.abs(frequencies))
        freq_domain[np.abs(frequencies) > threshold] = 0

        # Inverse FFT to restore time domain signal
        filtered_data[col] = np.real(np.fft.ifft(freq_domain))

    filtered_df = pd.DataFrame(filtered_data, index=df.index)

    # 3. Standardization
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(filtered_df)

    return scaled_data, scaler


if __name__ == "__main__":
    # Initialize model and deploy to NVIDIA GPU
    seq_length = 12
    model = WeatherCNN(num_features=4, seq_length=seq_length).to(device)

    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(
        "Model creation completed, successfully mounted NV GPU acceleration module, ready to receive FFT-processed 5-min aligned data.")

    # Simulate raw device collected data with noise and misaligned time
    num_samples = 100
    timestamps = pd.date_range('2023-01-01', periods=num_samples, freq='4T13S')  # Irregular timestamps

    # Pressure(Pa), wind speed(m/s), temperature(C), humidity(%RH)
    raw_pressure = 101325 + np.random.normal(0, 50, num_samples)
    raw_wind = 3.0 + np.random.normal(0, 1.5, num_samples)
    raw_temp = 20.0 + np.random.normal(0, 5.0, num_samples)
    raw_humidity = 60.0 + np.random.normal(0, 10.0, num_samples)

    features_data = np.column_stack((raw_pressure, raw_wind, raw_temp, raw_humidity))

    # FFT preprocessing and alignment
    processed_data, _ = preprocess_and_fft_align(timestamps, features_data)

    # Build sequence data
    x_seqs = []
    y_seqs = []
    for i in range(len(processed_data) - seq_length):
        x_seqs.append(processed_data[i:i + seq_length])
        # Use the temperature of the next time step (index 2) as prediction target
        y_seqs.append(processed_data[i + seq_length, 2])

    # Convert to PyTorch tensors
    # CNN1D input shape needs to be (batch_size, num_features, seq_length)
    X = torch.tensor(np.array(x_seqs), dtype=torch.float32).permute(0, 2, 1).to(device)
    Y = torch.tensor(np.array(y_seqs), dtype=torch.float32).view(-1, 1).to(device)

    print(f"Data processing completed, X shape: {X.shape}, Y shape: {Y.shape}")

    # Stage-by-stage Residual/Cascade Training
    # Requirements: Train each layer for 200 epochs, next layer learns from previous error, total 3 layers
    num_stages = 3
    epochs_per_stage = 200
    models = []

    # Current target starts as the original Y. In later stages, it will be the residual error.
    current_target = Y.clone()

    for stage in range(num_stages):
        print(f"\n--- Stage {stage + 1}/{num_stages}: Training {'Base' if stage == 0 else 'Residual Error'} Model ---")

        # Initialize a new model for this stage
        current_model = WeatherCNN(num_features=4, seq_length=seq_length).to(device)
        optimizer = optim.Adam(current_model.parameters(), lr=0.001)

        current_model.train()
        pbar = tqdm(range(epochs_per_stage), desc=f"Stage {stage + 1}")
        for epoch in pbar:
            optimizer.zero_grad()
            predictions = current_model(X)
            loss = criterion(predictions, current_target)
            loss.backward()
            optimizer.step()

            # Update the progress bar to show the target training loss dynamically
            pbar.set_postfix({'MSE Loss': f"{loss.item():.4f}"})

        # Calculate new residuals (what the current model failed to predict)
        # The next stage will use this 'residual error' as its target
        with torch.no_grad():
            current_model.eval()
            stage_pred = current_model(X)
            # Use in-place subtraction to save memory, important for Docker memory limits
            current_target.sub_(stage_pred)

        models.append(current_model)

        # Clear cache to prevent Docker Out-Of-Memory (OOM) Segfaults
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Combine predictions from all 3 layers and evaluate overall cascade performance
    with torch.no_grad():
        final_combined_predictions = torch.zeros_like(Y)
        for m in models:
            m.eval()
            final_combined_predictions += m(X)

        final_loss = criterion(final_combined_predictions, Y)

        # Calculate extra metrics for visibility
        test_rmse = torch.sqrt(final_loss)
        test_mae = torch.abs(final_combined_predictions - Y).mean()

        target_mean = torch.mean(Y)
        ss_tot = torch.sum((Y - target_mean) ** 2)
        ss_res = torch.sum((Y - final_combined_predictions) ** 2)
        r2_score = 1 - (ss_res / ss_tot)

    print(f"\n================ CASCADE TRAINING COMPLETED ================")
    print(f"Total Stages (Layers) Trained : {num_stages}")
    print(f"Final Cascade MSE Loss        : {final_loss.item():.4f}")
    print(f"Final Cascade RMSE Loss       : {test_rmse.item():.4f}")
    print(f"Final Cascade MAE Loss        : {test_mae.item():.4f}")
    print(f"Final R² Score ('Accuracy')   : {r2_score.item() * 100:.2f}%")
    print("============================================================")

    # Save all models together
    save_path = 'cascade_weather_model.pth'
    save_dict = {f'model_stage_{i+1}_state_dict': m.state_dict() for i, m in enumerate(models)}
    save_dict['num_stages'] = len(models)

    torch.save(save_dict, save_path)
    print(f"Cascade models successfully saved to {save_path}")
