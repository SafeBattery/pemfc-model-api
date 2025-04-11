from flask import Flask, request, jsonify
from model import load_model, predict_pw
from explain import explain_with_dynamic_mask
import numpy as np

app = Flask(__name__)

# 모델 불러오기
model = load_model("app/model_cpu.pth")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_sequence = np.array(data["input"])   # 2D 배열 (600 x 9)
    target_value = float(data["target"])       # 실제 PW 값 (예: 0.75)

    # 예측
    prediction = predict_pw(model, input_sequence)

    # 해석
    mask = explain_with_dynamic_mask(model, input_sequence, target_value)

    # 결과 반환
    return jsonify({
        "prediction": prediction,
        "mask": mask.tolist()  # numpy 배열은 json으로 변환 위해 list로
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
