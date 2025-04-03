import requests
import numpy as np

# 테스트용 더미 입력 (600 x 9)
dummy_input = np.random.rand(600, 9).tolist()

# 요청 데이터
data = {
    "input": dummy_input,
    "target": 0.75  # 아무 숫자나 테스트용으로
}

# Flask 서버로 POST 요청 보내기
res = requests.post("http://localhost:5000/predict", json=data)

# 결과 출력
print(res.status_code)
print(res.json())
