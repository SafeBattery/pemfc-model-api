# ── 1단계: 빌드 스테이지 ──
FROM python:3.10-slim AS builder

# 빌드에만 필요한 패키지 설치
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
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

WORKDIR /app
COPY requirements.txt .

# 휠 파일만 모아두기
RUN pip install --upgrade pip \
 && pip wheel --no-cache-dir --no-deps -w /wheels -r requirements.txt

# ── 2단계: 런타임 스테이지 ──
FROM python:3.10-slim

# 런타임에만 필요한 라이브러리 설치
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libatlas-base-dev \
      libffi-dev \
      libssl-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 빌드 스테이지에서 만든 휠만 설치
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# 소스 복사
COPY . .

EXPOSE 80

# Flask 실행
CMD ["python", "app/main.py"]
