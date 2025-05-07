import torch
import numpy as np
from UtotV_and_PW_model_api.Model.Informer import Informer

# ✅ 디바이스 설정 (전역 디폴트)
default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ 현재 디바이스:", default_device)

# ✅ 모델 불러오기 함수
def load_model(model_path="model.pth", device=None):
    if device is None:
        device = default_device

    model = Informer(
        input_size=9,
        output_size=2,
        d_model=128,
        n_heads=4,
        e_layers=2
    ).to(device)

    # 모델 가중치 불러오기
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ UtotV & PW 모델 로드 성공")
    return model

# ✅ 예측 함수
def predict(model, input_data, device=None):
    if device is None:
        device = default_device

    input_tensor = torch.FloatTensor(np.array(input_data)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    return output.squeeze().cpu().tolist()
