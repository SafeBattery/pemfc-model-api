# /opt/airflow/scripts/monitor_PWU.py
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

# Flask 모델 참조
sys.path.append('/opt/airflow/flask')
from Model.T3.Informer import Informer

# 설정
CSV_PATH = '/opt/airflow/data/data.csv'
MODEL_PATH = '/opt/airflow/data/models/PWU/model.pth'
RESULT_PATH = '/opt/airflow/data/monitor_PWU_result.json'

INPUT_COLS = ['U_totV', 'iA', 'P_H2_supply', 'P_H2_inlet', 'P_Air_supply',
              'P_Air_inlet', 'm_Air_write', 'm_H2_write', 'T_Stack_inlet']
TARGET_COLS = ['U_totV', 'PW']

WINDOW_SIZE = 600
TARGET_DISTANCE = 100
ONE_SEQ_LENGTH = 6000
BATCH_SIZE = 32
THRESHOLD = 0.02

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로드 및 정규화
df = pd.read_csv(CSV_PATH)
df = df.iloc[:31000]
scaler = MinMaxScaler()
df[INPUT_COLS + TARGET_COLS] = scaler.fit_transform(df[INPUT_COLS + TARGET_COLS])

test = df.iloc[20000:31000]
X = test[INPUT_COLS].values
y = test[TARGET_COLS].values

# 시퀀스 자르기
def slice_sequence(data, y, window_size, label_distance, seq_len):
    Xs, ys, i = [], [], 0
    while i + window_size + label_distance < len(data):
        Xs.append(data[i:i+window_size])
        ys.append(y[i+window_size+label_distance])
        if (i + window_size + label_distance + 1) % seq_len == 0:
            i += window_size + label_distance + 1
        else:
            i += 1
    return torch.FloatTensor(np.array(Xs)), torch.FloatTensor(np.array(ys))

X_tensor, y_tensor = slice_sequence(X, y, WINDOW_SIZE, TARGET_DISTANCE, ONE_SEQ_LENGTH)
loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=BATCH_SIZE)

# 모델 로드
model = Informer(input_size=len(INPUT_COLS), output_size=len(TARGET_COLS)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 평가
loss_fn = nn.MSELoss()
losses = []

with torch.no_grad():
    for xb, yb in loader:
        pred = model(xb.to(device))
        loss = loss_fn(pred, yb.to(device))
        losses.append(loss.item())

avg_loss = np.mean(losses)
print(f"[monitor_PWU] Test Loss: {avg_loss:.4f}")

# 결과 저장
with open(RESULT_PATH, 'w') as f:
    json.dump({
        "loss": float(avg_loss),
        "threshold": float(THRESHOLD),
        "retrain": bool(avg_loss > THRESHOLD)
    }, f)

print(f"[monitor_PWU] Result saved → {RESULT_PATH}")
