import requests
import numpy as np

url = "http://localhost:5000/predict"
data = np.tile(np.arange(0.1, 1.0, 0.1), (600, 1)).tolist()  # [600, 9] 입력

payload = {
    "input": data,
    "type": "PWU",
    "threshold": 0.05
}

res = requests.post(url, json=payload)
print(res.json())
