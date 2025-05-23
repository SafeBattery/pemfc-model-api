from flask import Flask, request, jsonify
import torch
import numpy as np
import joblib

pwu_scaler = joblib.load("/scaler/y_pwu_scaler.pkl")
t3_scaler = joblib.load("/scaler/y_t3_scaler.pkl")

app = Flask(__name__)
models = {
    "PWU": None,
    "T3": None
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

@app.route("/", methods=["GET"])
def index():
    return "✅ Flask API is running!", 200

import os
@app.route('/reload_model', methods=['POST'])
def reload_model():
    try:
        model_path = request.json['model_path']
        model_type = request.json.get('type', 'PWU')
        print(f"📦 요청받은 모델 타입: {model_type}, 경로: {model_path}")

        # 🔧 상대경로일 경우 절대경로로 변환
        if not os.path.isabs(model_path):
            model_path = os.path.join("/models", model_path)

        # 🔁 모델 클래스 분기 import
        if model_type == "PWU":
            from Model.PWU import Informer
            model = Informer(input_size=9, output_size=2, d_model=64, n_heads=8, e_layers=3)
        elif model_type == "T3":
            from Model.T3 import Informer
            model = Informer(input_size=4, output_size=1)
        else:
            return jsonify({"error": f"Unknown model type: {model_type}"}), 400

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models[model_type] = model
        return jsonify({"status": "success", "message": f"{model_type} model reloaded."})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("[DEBUG 1] 요청 수신됨")
        data = request.json['input']
        model_type = request.json.get('type', 'PWU')
        threshold = request.json.get('threshold', 0.02)
        target_index = request.json.get('target_index', None)

        print(f"[DEBUG 2] model_type: {model_type}")
        print(f"[DEBUG 3] data shape: {np.array(data).shape}")

        if model_type not in models or models[model_type] is None:
            return jsonify({"error": f"{model_type} 모델이 로드되지 않았습니다."}), 500

        # ✅ 입력 스케일링
        if model_type == "PWU":
            x_scaler = joblib.load("/scaler/x_pwu_scaler.pkl")
            input_array = np.array(data)
            inverse_indices = [0, 2, 3, 4, 5, 6, 7, 8]
            to_scale = input_array[:, inverse_indices]
            scaled_part = x_scaler.transform(to_scale)
            iA_diff = input_array[:, 1].reshape(-1, 1)
            scaled_input = np.concatenate([
                scaled_part[:, [0]],
                iA_diff,
                scaled_part[:, 1:]
            ], axis=1)
            input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0).to(device)

        elif model_type == "T3":
            x_scaler = joblib.load("/scaler/x_t3_scaler.pkl")
            input_array = np.array(data)
            scaled_input = x_scaler.transform(input_array)
            input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0).to(device)
        else:
            return jsonify({"error": f"{model_type}에 대한 입력 스케일러가 없습니다."}), 500

        model = models[model_type]

        # ✅ 예측
        with torch.no_grad():
            output = model(input_tensor).squeeze().cpu()
            if isinstance(output, torch.Tensor) and output.dim() == 0:
                pred = output.item()
            else:
                pred = output.tolist()

        print(f"[DEBUG 4] pred: {pred} ({type(pred)})")

        # ✅ tensor → float 변환 함수 정의
        def to_scalar(x):
            if isinstance(x, torch.Tensor):
                if x.dim() == 0:
                    return x.item()
                elif x.dim() == 1 and x.numel() == 1:
                    return x[0].item()
                else:
                    raise ValueError("예상치 못한 tensor 차원입니다.")
            elif isinstance(x, list) and len(x) == 1:
                return float(x[0])
            elif isinstance(x, (float, int)):
                return float(x)
            else:
                raise ValueError("지원되지 않는 pred 형식입니다.")

        # ✅ 역변환 함수
        def inverse_scale(pred, model_type):
            if model_type == "PWU":
                arr = np.array(pred).reshape(1, -1)
                return pwu_scaler.inverse_transform(arr).flatten().tolist()
            elif model_type == "T3":
                val = to_scalar(pred)
                return t3_scaler.inverse_transform(np.array([[val]])).flatten().tolist()
            else:
                return pred

        original_pred = inverse_scale(pred, model_type)

        result = {
            "prediction": pred,
            "original_prediction": original_pred,
            "threshold": threshold
        }

        # ✅ 조건 위반 판단
        trigger_explain = False
        violated_index = None

        if model_type == "PWU" and isinstance(pred, list) and len(pred) >= 2:
            if not (0.628 <= pred[0] <= 0.941):
                trigger_explain = True
                violated_index = 0
            elif not (0.221 <= pred[1] <= 0.698):
                trigger_explain = True
                violated_index = 1

        elif model_type == "T3":
            pred_val = to_scalar(pred)
            print(f"[DEBUG 5] pred_val: {pred_val}")
            if not (0.336 <= pred_val <= 0.983):
                trigger_explain = True
                violated_index = 0

        # ✅ Explain 호출
        if trigger_explain:
            print(f"[DEBUG 6] Explain triggered. violated_index: {violated_index}")
            input_seq = torch.FloatTensor(np.array(data))
            if isinstance(pred, (float, int)):
                target_tensor = torch.tensor([pred], dtype=torch.float32)
            elif isinstance(pred, list):
                target_tensor = torch.tensor(pred, dtype=torch.float32).reshape(-1)
            elif isinstance(pred, torch.Tensor):
                target_tensor = pred.float().reshape(-1)
            else:
                raise ValueError("지원되지 않는 pred 형식입니다.")

            mask = explain(model, input_seq, target_tensor, violated_index)
            result["mask"] = mask.tolist()
            result["explained_target_index"] = violated_index

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Flask ERROR] 예측 중 오류 발생: {str(e)}")
        return jsonify({"error": str(e)}), 500



from explain import explain
import numpy as np

@app.route('/predict_and_explain', methods=['POST'])
def predict_and_explain():
    try:
        data = request.json['input']          # [600, feature_dim]
        target = request.json['target']       # 예측 정답 값
        model_type = request.json.get('type', 'PWU')
        target_index = request.json.get('target_index', None)

        if model_type not in models or models[model_type] is None:
            return jsonify({"error": f"{model_type} 모델이 로드되지 않았습니다."}), 500

        model = models[model_type]

        # 입력값 → 텐서 변환
        input_seq = torch.FloatTensor(np.array(data))
        target_tensor = torch.FloatTensor(np.array(target))

        # 마스크 해석 실행
        mask = explain(model, input_seq, target_tensor, target_index)

        return jsonify({"mask": mask.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 자동 모델 로드
import os
if os.path.exists("/models/PWU/model.pth"):
    from Model.PWU import Informer
    model = Informer(input_size=9, output_size=2, d_model=64, n_heads=8, e_layers=3)
    model.load_state_dict(torch.load("/models/PWU/model.pth", map_location='cpu'))
    model.eval()
    models["PWU"] = model
    print("✅ [Flask] PWU model loaded at startup")

if os.path.exists("/models/T3/model.pth"):
    from Model.T3 import Informer
    model = Informer(input_size=4, output_size=1)
    model.load_state_dict(torch.load("/models/T3/model.pth", map_location='cpu'))
    model.eval()
    models["T3"] = model
    print("✅ [Flask] T3 model loaded at startup")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
