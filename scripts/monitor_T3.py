import sys
import os
import json
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ✅ Airflow 환경에서 flask 디렉토리를 모듈 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'flask'))

# ✅ 모델 임포트
from Model.T3.Informer import Informer  # T3 모델 클래스 정의되어 있는 파일

# ✅ 설정
WINDOW_SIZE = 600
TARGET_DISTANCE = 100
ONE_SEQ_LENGTH = 6000
BATCH_SIZE = 20
THRESHOLD = 0.02
MODEL_PATH = '/opt/airflow/data/models/T3/model.pth'
RESULT_PATH = '/opt/airflow/data/monitor_T3_result.json'
INPUT_COLS = ['P_H2_inlet', 'P_Air_inlet', 'T_Heater', 'T_Stack_inlet']
TARGET_COL = 'T_3'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 시퀀스 슬라이싱 함수
def slice_sequence(data, y, window_size, label_distance, seq_len):
    X, Y, i = [], [], 0
    while i + window_size + label_distance < len(data):
        X.append(data[i:i + window_size])
        Y.append(y[i + window_size + label_distance])
        if (i + window_size + label_distance + 1) % seq_len == 0:
            i += window_size + label_distance + 1
        else:
            i += 1
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(Y))

# ✅ 데이터 로드 및 정규화
df = pd.read_csv('/opt/airflow/data/full_test_data.csv')
df = df.iloc[::5].reset_index(drop=True)

scaler = MinMaxScaler()
df[INPUT_COLS + [TARGET_COL]] = scaler.fit_transform(df[INPUT_COLS + [TARGET_COL]])

test_df = df.iloc[:3000]
test_X, test_y = slice_sequence(
    np.array(test_df[INPUT_COLS]),
    np.array(test_df[TARGET_COL]),
    WINDOW_SIZE, TARGET_DISTANCE, ONE_SEQ_LENGTH
)

loader = DataLoader(TensorDataset(test_X, test_y), batch_size=BATCH_SIZE, shuffle=False)

# ✅ 모델 정의 및 로드
model = Informer(input_size=len(INPUT_COLS), output_size=1, d_model=128, n_heads=4, e_layers=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ✅ 성능 측정
loss_fn = nn.MSELoss()
losses = []
with torch.no_grad():
    for xb, yb in loader:
        pred = model(xb.to(device))
        yb = yb.unsqueeze(1).to(device)
        loss = loss_fn(pred, yb)
        losses.append(loss.item())

avg_loss = np.mean(losses)
print(f"[monitor_PWU] Test Loss: {avg_loss:.4f}")

with open(RESULT_PATH, 'w') as f:
    json.dump({
        "loss": float(avg_loss),
        "threshold": THRESHOLD,
        "retrain": bool(avg_loss > THRESHOLD)
    }, f)

