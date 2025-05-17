import torch
import numpy as np

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ 현재 디바이스:", default_device)

# ✅ 모델 로딩 함수
def load_model(model_type="PWU", model_path=None, device=None):
    if device is None:
        device = default_device

    if model_type == "PWU":
        from Model.PWU import Informer
        input_size = 9
        output_size = 2
        default_path = "models/PWU.pth"
    elif model_type == "T3":
        from Model.T3 import Informer
        input_size = 4
        output_size = 1
        default_path = "models/T3.pth"
    else:
        raise ValueError(f"❌ Unknown model type: {model_type}")

    if model_path is None:
        model_path = default_path

    model = Informer(
        input_size=input_size,
        output_size=output_size,
        d_model=128,
        n_heads=4,
        e_layers=2
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# ✅ 예측 함수
def predict(model, input_data, device=None):
    if device is None:
        device = default_device

    input_tensor = torch.FloatTensor(np.array(input_data)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    return output.squeeze().cpu().tolist()
