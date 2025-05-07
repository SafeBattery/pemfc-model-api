from flask import Flask
from T3_model_api.main import load_model as load_T3_model
from UtotV_and_PW_model_api.main import load_model as load_UtotV_and_PW_model
import torch

from views.t3 import predict_and_explain_T3
from views.utotv_and_pw import predict_and_explain_UtotV_and_PW


# ✅ 디바이스 설정 (GPU 우선)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ 현재 디바이스:", device)

# ✅ 모델 로드
model_T3 = load_T3_model(model_path="raw_models/model_T3.pth", device=device)
model_UtotV_and_PW = load_UtotV_and_PW_model(model_path="raw_models/model.pth", device=device)

# ✅ Flask 앱 생성
app = Flask(__name__)

@app.route("/predict_and_explain/T3", methods=["POST"])
def foo():
    # todo: 이름 바꾸기
    return predict_and_explain_T3(model=model_T3, device=device)


@app.route("/predict_and_explain/UtotV_and_PW", methods=["POST"])
def foo2():
    # todo: 이름 바꾸기
    return predict_and_explain_UtotV_and_PW(model=model_UtotV_and_PW, device=device)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
