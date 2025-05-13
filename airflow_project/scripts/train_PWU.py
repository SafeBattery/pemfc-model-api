# /opt/airflow/scripts/train_PWU.py

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.append('/models')
from Model.PWU.Informer import Informer

# 설정
CSV_PATH = "/opt/airflow/data/data.csv"
MODEL_PATH = "/models/PWU/model.pth"
INPUT_COLS = ['U_totV', 'iA', 'P_H2_supply', 'P_H2_inlet', 'P_Air_supply',
              'P_Air_inlet', 'm_Air_write', 'm_H2_write', 'T_Stack_inlet']
TARGET_COLS = ['U_totV', 'PW']

WINDOW_SIZE = 600
TARGET_DISTANCE = 100
ONE_SEQ_LENGTH = 6000
BATCH_SIZE = 32
EPOCHS = 1
LR = 5e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로딩
df = pd.read_csv(CSV_PATH)
df = df.iloc[:31000]
scaler = MinMaxScaler()
df[INPUT_COLS + TARGET_COLS] = scaler.fit_transform(df[INPUT_COLS + TARGET_COLS])

train = df.iloc[0:20000]
X = train[INPUT_COLS].values
y = train[TARGET_COLS].values


# 시퀀스 자르기
def slice_sequence(data, y, window_size, label_distance, seq_len):
    Xs, ys, i = [], [], 0
    while i + window_size + label_distance < len(data):
        Xs.append(data[i:i + window_size])
        ys.append(y[i + window_size + label_distance])
        if (i + window_size + label_distance + 1) % seq_len == 0:
            i += window_size + label_distance + 1
        else:
            i += 1
    return torch.FloatTensor(np.array(Xs)), torch.FloatTensor(np.array(ys))


X_tensor, y_tensor = slice_sequence(X, y, WINDOW_SIZE, TARGET_DISTANCE, ONE_SEQ_LENGTH)
loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=BATCH_SIZE, shuffle=True)

# 모델 학습
model = Informer(input_size=len(INPUT_COLS), output_size=len(TARGET_COLS)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for xb, yb in loader:
        pred = model(xb.to(device))
        loss = criterion(pred, yb.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[Epoch {epoch + 1}] Loss: {total_loss / len(loader):.4f}")

# 모델 저장
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"[✅ Saved] Trained model → {MODEL_PATH}")
