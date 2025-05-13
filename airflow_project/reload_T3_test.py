# reload_T3_test.py
import requests

res = requests.post("http://localhost:5000/reload_model", json={
    "model_path": "/models/T3/model.pth",
    "type": "T3"
})

print("재로딩 응답:", res.json())
