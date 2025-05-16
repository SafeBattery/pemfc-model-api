import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

# 경로 설정
sys.path.append('/models')
from Model.T3.Informer import Informer  # T3용 Informer 모듈

# 설정
CSV_PATH = "/opt/airflow/data/data.csv"
MODEL_PATH = "/models/T3/model.pth"

INPUT_COLS = ['P_H2_inlet', 'P_Air_inlet', 'T_Heater', 'T_Stack_inlet']
TARGET_COL = 'T_3'

WINDOW_SIZE = 600
TARGET_DISTANCE = 100
ONE_SEQ_LENGTH = 6000
BATCH_SIZE = 32
EPOCHS = 1
LR = 5e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 슬라이싱 함수
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


# ✅ 데이터 로드 및 전처리
df = pd.read_csv(CSV_PATH)
df = df.iloc[::5].reset_index(drop=True)  # 다운샘플링

scaler = MinMaxScaler()
df[INPUT_COLS + [TARGET_COL]] = scaler.fit_transform(df[INPUT_COLS + [TARGET_COL]])

train_df = df.iloc[3000:]
train_X, train_y = slice_sequence(
    np.array(train_df[INPUT_COLS]),
    np.array(train_df[TARGET_COL]),
    WINDOW_SIZE,
    TARGET_DISTANCE,
    ONE_SEQ_LENGTH
)

loader = DataLoader(TensorDataset(train_X, train_y), batch_size=BATCH_SIZE, shuffle=True)

# ✅ 모델 정의 및 학습
model = Informer(input_size=len(INPUT_COLS), output_size=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for xb, yb in loader:
        pred = model(xb.to(device)).squeeze()  # [batch, 1] → [batch]
        loss = criterion(pred, yb.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[Epoch {epoch + 1}] Loss: {total_loss / len(loader):.4f}")

# ✅ 모델 저장
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"[✅ Saved] Trained model → {MODEL_PATH}")
