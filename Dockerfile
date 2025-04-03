FROM python:3.10-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    liblzma-dev \
    libsqlite3-dev \
    zlib1g-dev \
    libatlas-base-dev \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app
COPY . .

# pip 최신화 및 패키지 설치
RUN pip install --upgrade pip
RUN pip install --no-cache-dir flask==2.2.5 torch==1.13.1 numpy==1.21.6 \
    matplotlib==3.5.3 scikit-learn==1.0.2 requests==2.31.0 seaborn==0.11.2

# Flask 앱 실행
CMD ["python", "app/main.py"]

