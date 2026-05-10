import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from weather_prediction_model import WeatherCNN, preprocess_and_fft_align, device

def train_cascade_models(X_train, Y_train, seq_length=12, num_stages=3, epochs_per_stage=200):
    print(f"\n[Training] Starting Cascade Fusion Training ({num_stages} stages)...")
    criterion = nn.MSELoss()
    models = []
    current_target = Y_train.clone()

    for stage in range(num_stages):
        print(f"--- Stage {stage + 1}: Training {'Base' if stage == 0 else 'Residual'} Model ---")
        current_model = WeatherCNN(num_features=4, seq_length=seq_length).to(device)
        optimizer = optim.Adam(current_model.parameters(), lr=0.001)

        current_model.train()

        # Add a tqdm progress bar for epochs in each stage
        pbar = tqdm(range(epochs_per_stage), desc=f"Stage {stage + 1}")
        for epoch in pbar:
            optimizer.zero_grad()
            predictions = current_model(X_train)
            loss = criterion(predictions, current_target)
            loss.backward()
            optimizer.step()

            # Update tqdm progress bar with the current loss
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        # Update target to the remaining residual
        with torch.no_grad():
            current_model.eval()
            stage_pred = current_model(X_train)
            current_target = current_target - stage_pred

        models.append(current_model)

    print("[Training] Completed. Saving models...")
    # Save the models
    save_path = 'cascade_weather_model.pth'
    save_dict = {f'model_stage_{i+1}_state_dict': m.state_dict() for i, m in enumerate(models)}
    save_dict['num_stages'] = len(models)
    torch.save(save_dict, save_path)
    print(f"-> Models saved to {save_path}")

    return models

def test_model():
    print("Loading data for training & testing...")

    # 1. Load training data (data.txt)
    try:
        df_train = pd.read_csv('./data.txt')
        train_data = df_train.values
        print(f"Successfully loaded {len(train_data)} train samples from ./data.txt")
    except Exception as e:
        print(f"Error loading training data: {e}")
        return

    # 2. Load testing data (testdata.txt)
    try:
        df_test = pd.read_csv('./testdata.txt')
        test_data = df_test.values
        print(f"Successfully loaded {len(test_data)} test samples from ./testdata.txt")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    # Helper function to build sequence data
    def build_sequences(data_array, seq_length=12):
        x_seqs, y_seqs = [], []
        for i in range(len(data_array) - seq_length):
            x_seqs.append(data_array[i:i+seq_length])
            y_seqs.append(data_array[i+seq_length, 2])

        X = torch.tensor(np.array(x_seqs), dtype=torch.float32).permute(0, 2, 1).to(device)
        Y = torch.tensor(np.array(y_seqs), dtype=torch.float32).view(-1, 1).to(device)
        return X, Y

    seq_length = 12
    X_train, Y_train = build_sequences(train_data, seq_length)
    X_test, Y_test = build_sequences(test_data, seq_length)

    print(f"Data split: {len(X_train)} train samples, {len(X_test)} test samples.")

    # 1. Execute full cascade training and save model
    cascade_models = train_cascade_models(X_train, Y_train, seq_length=seq_length)

    # 2. Test the cascaded models on strictly unseen testing data
    print("\nStarting model testing on test set with cascade fusion...")
    criterion = nn.MSELoss()
    with torch.no_grad():
        test_predictions = torch.zeros_like(Y_test)

        for m in cascade_models:
            m.eval()
            test_predictions += m(X_test)

        # Calculate comprehensive metrics
        test_mse = criterion(test_predictions, Y_test)
        test_rmse = torch.sqrt(test_mse)
        test_mae = torch.abs(test_predictions - Y_test).mean()

        # Calculate R-squared (R²) acting as "Accuracy" for regression
        target_mean = torch.mean(Y_test)
        ss_tot = torch.sum((Y_test - target_mean) ** 2)
        ss_res = torch.sum((Y_test - test_predictions) ** 2)
        r2_score = 1 - (ss_res / ss_tot)

        print("\n================ EVALUATION METRICS ================")
        print(f"Test MSE (Mean Squared Error)      : {test_mse.item():.4f}")
        print(f"Test RMSE (Root Mean Squared Error): {test_rmse.item():.4f}")
        print(f"Test MAE (Mean Absolute Error)     : {test_mae.item():.4f}")
        print(f"Test R² Score ('Accuracy' metric)  : {r2_score.item() * 100:.2f}%")
        print("====================================================")

        # Show prediction examples
        print("\nSample predictions vs Ground truth:")
        for i in range(min(5, len(Y_test))):
            pred_val = test_predictions[i].item()
            true_val = Y_test[i].item()
            print(f"  Sample {i+1}: Prediction = {pred_val:.4f}, Ground Truth = {true_val:.4f}")

if __name__ == "__main__":
    test_model()
