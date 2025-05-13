import requests
import numpy as np

# 테스트 입력 (600, 4)
X_test = np.tile(np.linspace(0.2, 0.8, 4), (600, 1)).tolist()

res = requests.post("http://localhost:5000/predict", json={
    "input": X_test,
    "type": "T3"
})

print("응답 결과:", res.json())
