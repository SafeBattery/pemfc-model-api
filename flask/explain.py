import torch
import numpy as np
from attribution.mask import Mask
from attribution.perturbation import FadeMovingAverageWindow
from Utils.losses import mse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ [Explain] 디바이스 상태:", device)

def explain(model, input_seq, target_tensor, target_index=None):
    """
    - model: 학습된 PyTorch 모델
    - input_seq: torch.FloatTensor, shape: [seq_len, feature]
    - target_tensor: torch.FloatTensor, shape: [output_dim] or scalar
    - target_index: 다중 출력인 경우 사용할 인덱스 (PWU용: 0 or 1)
    """
    pert = FadeMovingAverageWindow(device, window_size=600)
    mask = Mask(perturbation=pert, device=device, task="regression", verbose=False, deletion_mode=True)

    def f_target(x):
        model.train()
        out = model(x.unsqueeze(0)).squeeze()
        if out.dim() == 0 or target_index is None:
            return out
        elif out.dim() == 1 and target_index < out.size(0):
            return out[target_index]
        else:
            raise ValueError("예상치 못한 출력 형태 또는 인덱스 초과입니다.")

    # ✅ 타겟값 정의
    if target_index is None:
        target_value = target_tensor.item()
    else:
        target_value = target_tensor[target_index].item()

    # ✅ 마스크 학습
    mask.fit(
        X=input_seq.to(device),
        f=f_target,
        loss_function=mse,
        target=target_value,
        learning_rate=1,
        size_reg_factor_init=0.1,
        size_reg_factor_dilation=200,
        initial_mask_coeff=0.7,
        n_epoch=300,
        momentum=0.8,
        time_reg_factor=0.2,
    )

    return mask.mask_tensor.detach().cpu().numpy()
