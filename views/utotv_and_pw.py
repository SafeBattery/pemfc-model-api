from flask import request, jsonify
from UtotV_and_PW_model_api.main import predict
from UtotV_and_PW_model_api.explain import explain
import torch
import numpy as np


def predict_and_explain_UtotV_and_PW(model, device):
    try:
        data = request.get_json()

        # ✅ 입력 처리
        input_data = np.array(data["input"], dtype=np.float32)
        threshold = data.get("threshold", [0.0, 0.0])

        # ✅ 예측 수행
        prediction = predict(model, input_data, device=device)
        explanations = {}
        triggered = False

        # ✅ 임계값 초과 여부 판단 + 마스크 해석
        for i, (value, th) in enumerate(zip(prediction, threshold)):
            if abs(value) > th:
                input_tensor = torch.FloatTensor(input_data).to(device)
                target_tensor = torch.FloatTensor(prediction).to(device)
                mask = explain(
                    model=model,
                    input_seq=input_tensor,
                    target_tensor=target_tensor,
                    target_index=i
                )
                explanations[f"mask_{i}"] = mask.tolist()
                triggered = True

        return jsonify({
            "prediction": prediction,
            "explanation": explanations if triggered else "No threshold exceeded."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
