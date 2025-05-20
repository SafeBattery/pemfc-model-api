# ✅ 1. Python 기반 이미지
FROM python:3.9-slim

# ✅ 2. 작업 디렉토리 설정
WORKDIR /app

# ✅ 3. CUDA 지원 PyTorch 먼저 설치 (예: CUDA 11.8)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ✅ 4. requirements.txt 복사 및 설치 (torch 제거 필수)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ✅ 5. Flask 앱 전체 복사
COPY . .

# ✅ 6. models 경로 명시적으로 생성
RUN mkdir -p /models

# ✅ 7. Flask 서버 실행
CMD ["python", "app.py"]
