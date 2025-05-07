import torch
import numpy as np
from UtotV_and_PW_model_api.attribution.mask import Mask
from UtotV_and_PW_model_api.attribution.perturbation import FadeMovingAverageWindow
from UtotV_and_PW_model_api.Utils.losses import mse

# ✅ 디바이스 자동 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ [Explain] 디바이스 상태:", device)

# ✅ 해석 함수
def explain(model, input_seq, target_tensor, target_index=0):
    """
    - model: 학습된 PyTorch 모델
    - input_seq: [seq_len, feature] 형태의 torch.FloatTensor
    - target_tensor: [output_dim] 형태의 torch.FloatTensor
    - target_index: 예측할 대상 인덱스 (0 or 1)
    """

    # ✅ Perturbation 설정
    pert = FadeMovingAverageWindow(device, window_size=600)

    # ✅ 마스크 객체 생성
    mask = Mask(
        perturbation=pert,
        device=device,
        task="regression",
        verbose=False,
        deletion_mode=True
    )

    # ✅ 예측 함수 정의
    def f_target(x):
        model.train()
        return model(x.unsqueeze(0)).squeeze()[target_index]

    # ✅ 마스크 학습
    mask.fit(
        X=input_seq.to(device),
        f=f_target,
        loss_function=mse,
        target=target_tensor[target_index].item(),
        learning_rate=1,
        size_reg_factor_init=0.1,
        size_reg_factor_dilation=200,
        initial_mask_coeff=0.7,
        n_epoch=300,
        momentum=0.8,
        time_reg_factor=0.2,
    )

    # ✅ 결과 반환
    return mask.mask_tensor.detach().cpu().numpy()
