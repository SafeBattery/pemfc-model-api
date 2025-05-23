import pandas as pd
import numpy as np
import requests
import json

# ✅ CSV 경로 설정
csv_path = r"C:\Users\kk\PycharmProjects\cap_airflow\data\data.csv"

input_columns = ['iA', 'iA_diff','P_H2_supply', 'P_H2_inlet', 'P_Air_supply',
           'P_Air_inlet', 'm_Air_write', 'm_H2_write', 'T_Stack_inlet']

# ✅ 읽어올 행 범위 지정
start_row = 28300
window_size = 600

# ✅ 데이터 로딩 및 입력 구성
df = pd.read_csv(csv_path)
input_data = df[input_columns].iloc[start_row:start_row + window_size].to_numpy().tolist()

# ✅ Flask에 요청 보낼 payload
payload = {
    "type": "PWU",
    "threshold": 0.05,
    "input": input_data
}

# ✅ 요청 전 디버깅 출력
print(f"[INFO] Sending data with shape: {np.array(input_data).shape}")

# ✅ 요청 보내기
url = "http://localhost:5000/predict"
response = requests.post(url, json=payload)

# ✅ 예측 결과 출력 이후에 정답 출력
if response.status_code == 200:
    result = response.json()
    print("[✅ SUCCESS] 응답 결과:")
    print(json.dumps(result, indent=2))

    # ✅ 정답값 출력: start_row + 100 위치의 PW, U_totV
    target_row = start_row + 700
    pw = df.loc[target_row, 'PW']
    u_totv = df.loc[target_row, 'U_totV']

    print(f"[🎯 정답] {target_row}번째 row - U_totV: {u_totv:.4f}, PW: {pw:.4f}")
else:
    print(f"[❌ ERROR] status code: {response.status_code}")
    print(response.text)

