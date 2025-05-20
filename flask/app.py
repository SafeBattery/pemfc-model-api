from flask import Flask, request, jsonify
import torch

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
            model = Informer(input_size=9, output_size=2)
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
        data = request.json['input']         # 입력 데이터: [600, feature_dim]
        model_type = request.json.get('type', 'PWU')
        threshold = request.json.get('threshold', 0.02)
        target_index = request.json.get('target_index', None)

        if model_type not in models or models[model_type] is None:
            return jsonify({"error": f"{model_type} 모델이 로드되지 않았습니다."}), 500

        model = models[model_type]
        input_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)  # shape: [1, 600, feature_dim]

        # ✅ 예측
        with torch.no_grad():
            pred = model(input_tensor).squeeze().cpu().tolist()

        result = {
            "prediction": pred,
            "threshold": threshold
        }

        # ✅ 조건 충족 시 explain 실행 (custom 범위로 설정)
        trigger_explain = False
        violated_index = None  # ✅ 범위 벗어난 인덱스 추적용

        if model_type == "PWU" and isinstance(pred, list) and len(pred) >= 2:
            # pred[0]: U_totV (0.628 ~ 0.941)
            # pred[1]: PW     (0.221 ~ 0.698)
            if not (0.628 <= pred[0] <= 0.941):
                trigger_explain = True
                violated_index = 0
            elif not (0.221 <= pred[1] <= 0.698):
                trigger_explain = True
                violated_index = 1

        elif model_type == "T3" and isinstance(pred, (float, int, list)):
            pred_val = pred if isinstance(pred, (float, int)) else pred[0]
            if not (0.336 <= pred_val <= 0.983):
                trigger_explain = True
                violated_index = 0

        if trigger_explain:
            input_seq = torch.FloatTensor(np.array(data))  # [600, feature_dim]
            target_tensor = torch.FloatTensor(np.array(pred))
            mask = explain(model, input_seq, target_tensor, violated_index)
            result["mask"] = mask.tolist()
            result["explained_target_index"] = violated_index  # 추가 정보 반환

        return jsonify(result)


    except Exception as e:

        print(f"[Flask ERROR] 예측 중 오류 발생: {str(e)}")  # 로그 출력 추가

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
    model = Informer(input_size=9, output_size=2)
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
