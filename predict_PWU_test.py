import requests
import numpy as np
import json

# ✅ 테스트용 T3 입력 생성: shape (600, 4)
X_test = np.tile(np.linspace(0.2, 0.8, 9), (600, 1)).tolist()

payload = {
    "type": "PWU",
    "threshold": 0.05,
    "input": X_test
}

# ✅ Flask 서버에 요청 (로컬에서 실행 중이면 localhost:5000, Docker 환경이면 flask-api:5000)
url = "http://localhost:5000/predict"  # or "http://flask-api:5000/predict" inside Docker

response = requests.post(url, json=payload)

if response.status_code == 200:
    result = response.json()
    print("[✅ SUCCESS] 응답 결과:")
    print(json.dumps(result, indent=2))
else:
    print(f"[❌ ERROR] status code: {response.status_code}")
    print(response.text)

# import requests
# import json
# import numpy as np
#
# # 예시 데이터: (600, 4)짜리 입력 생성
# fake_data = np.random.rand(600, 4).tolist()
#
# payload = {
#     "input": fake_data,
#     "type": "T3",
#     "threshold": 0.02
# }
#
# res = requests.post("http://localhost:5000/predict", json=payload)
# print("[✅ RESPONSE]", res.status_code)
# print(res.json())
