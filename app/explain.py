import torch
from attribution.mask import Mask
from attribution.perturbation import FadeMovingAverageWindow
from Utils.losses import log_loss
import numpy as np


def explain_with_dynamic_mask(model, input_sequence, target_value, window_size=600, area=0.35, epochs=1):
    device = torch.device("cpu")  # 명시적으로 CPU 사용
    model.to(device)
    model.train()  # 해석 시 train 모드 필요

    seqq = torch.FloatTensor(input_sequence).to(device)
    target_tensor = torch.tensor(target_value).to(device)

    def f(x):
        model.train()
        out = model(x.unsqueeze(0))
        return out.squeeze()

    # perturbation operator 정의
    pert = FadeMovingAverageWindow(device, window_size=window_size)
    mask = Mask(pert, device, task="regression", verbose=False, deletion_mode=True)

    # 마스크 학습을 통해 중요한 입력 부분 탐색
    mask.fit(
        X=seqq,
        f=f,
        loss_function=log_loss,
        keep_ratio=area,
        target=target_tensor,
        learning_rate=1,
        size_reg_factor_init=0.1,
        size_reg_factor_dilation=1000,
        initial_mask_coeff=0.4,
        n_epoch=epochs,
        momentum=0.9,
        time_reg_factor=1.0,
    )

    return mask.mask_tensor.detach().cpu().numpy()