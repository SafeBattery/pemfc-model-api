from flask import Flask, request, jsonify
import torch

app = Flask(__name__)
models = {
    "PWU": None,
    "T3": None
}
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

        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        models[model_type] = model
        return jsonify({"status": "success", "message": f"{model_type} model reloaded."})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['input']       # 예: [600, 9]
        model_type = request.json.get('type', 'PWU')
        threshold = request.json.get('threshold', 0.02)

        if model_type not in models or models[model_type] is None:
            return jsonify({"error": f"{model_type} 모델이 로드되지 않았습니다."}), 500

        model = models[model_type]
        input_tensor = torch.FloatTensor(data).unsqueeze(0)  # [1, 600, dim]

        with torch.no_grad():
            pred = model(input_tensor).squeeze().tolist()

        return jsonify({"prediction": pred, "threshold": threshold})
    except Exception as e:
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
