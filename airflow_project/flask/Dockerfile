# ✅ 1. Python 기반 이미지
FROM python:3.9-slim

# ✅ 2. 작업 디렉토리 설정
WORKDIR /app

# ✅ 3. requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ✅ 4. Flask 앱 전체 복사
COPY . .

# ✅ 5. models 경로 명시적으로 생성 (예외 방지)
RUN mkdir -p /models

# ✅ 6. Flask 서버 실행
CMD ["python", "app.py"]
