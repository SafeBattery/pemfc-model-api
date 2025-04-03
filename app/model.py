import torch
import torch.nn as nn
import numpy as np
from Model.LSTM_cpu import LSTM  # 학습 시 사용한 LSTM 모델 구조


def load_model(weight_path, input_size=9, hidden_size=64, num_layers=5, window_length=600):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        sequence_length=window_length,
        num_layers=num_layers
    )
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_pw(model, input_sequence):
    # input_sequence: 2D array (600 x 9)
    model.eval()
    windows = torch.FloatTensor(np.array(input_sequence))  # 단일 시퀀스 (600 x 9)
    input_tensor = windows.unsqueeze(0)  # (1 x 600 x 9)
    with torch.no_grad():
        output = model(input_tensor)
    return output.squeeze().cpu().item()
