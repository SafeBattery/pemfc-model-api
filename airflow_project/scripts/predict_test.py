import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from Model.T3.Informer import Informer
import matplotlib.pyplot as plt

# 설정
WINDOW_SIZE = 600
TARGET_DISTANCE = 100
SEQ_LENGTH = 6000
BATCH_SIZE = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 슬라이싱 함수
def slice_sequence(data, y, window_size, label_distance, seq_len):
    X, Y, i = [], [], 0
    while i + window_size + label_distance < len(data):
        X.append(data[i:i + window_size])
        Y.append(y[i + window_size + label_distance])
        if (i + window_size + label_distance + 1) % seq_len == 0:
            i += window_size + label_distance + 1
        else:
            i += 1
    return torch.FloatTensor(X), torch.FloatTensor(Y)

# ✅ PWU 예측
def predict_PWU():
    INPUT_COLS = ['U_totV', 'iA', 'P_H2_supply', 'P_H2_inlet', 'P_Air_supply',
                  'P_Air_inlet', 'm_Air_write', 'm_H2_write', 'T_Stack_inlet']
    TARGET_COLS = ['U_totV', 'PW']
    MODEL_PATH = 'models/PWU/model.pth'

    df = pd.read_csv('data/data.csv')
    scaler = MinMaxScaler()
    df[INPUT_COLS + TARGET_COLS] = scaler.fit_transform(df[INPUT_COLS + TARGET_COLS])

    test = df.iloc[:15000]
    X, y = slice_sequence(
        np.array(test[INPUT_COLS]), np.array(test[TARGET_COLS]),
        WINDOW_SIZE, TARGET_DISTANCE, SEQ_LENGTH
    )
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE)

    model = Informer(input_size=len(INPUT_COLS), output_size=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    preds, actuals = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device))
            preds.append(out.cpu().numpy())
            actuals.append(yb.numpy())

    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(actuals[:, 0], label="Actual U_totV")
    plt.plot(preds[:, 0], label="Pred U_totV")
    plt.legend(); plt.grid(); plt.title("PWU - U_totV")

    plt.subplot(1, 2, 2)
    plt.plot(actuals[:, 1], label="Actual PW")
    plt.plot(preds[:, 1], label="Pred PW")
    plt.legend(); plt.grid(); plt.title("PWU - PW")
    plt.tight_layout()
    plt.show()

# ✅ T3 예측
def predict_T3():
    INPUT_COLS = ['P_H2_inlet', 'P_Air_inlet', 'T_Heater', 'T_Stack_inlet']
    TARGET_COL = 'T_3'
    MODEL_PATH = 'models/T3/model.pth'

    df = pd.read_csv('data/full_test_data.csv')
    df = df.iloc[::5].reset_index(drop=True)  # 다운샘플링
    scaler = MinMaxScaler()
    df[INPUT_COLS + [TARGET_COL]] = scaler.fit_transform(df[INPUT_COLS + [TARGET_COL]])

    test = df.iloc[:3000]
    X, y = slice_sequence(
        np.array(test[INPUT_COLS]), np.array(test[TARGET_COL]),
        WINDOW_SIZE, TARGET_DISTANCE, SEQ_LENGTH
    )
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE)

    model = Informer(input_size=len(INPUT_COLS), output_size=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    preds, actuals = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device))
            preds.append(out.cpu().numpy())
            actuals.append(yb.unsqueeze(1).numpy())

    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)

    plt.figure(figsize=(8, 4))
    plt.plot(actuals, label="Actual T3")
    plt.plot(preds, label="Predicted T3")
    plt.title("T3 Prediction")
    plt.grid(); plt.legend(); plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("🔍 Predicting PWU...")
    predict_PWU()
    print("🔍 Predicting T3...")
    predict_T3()
