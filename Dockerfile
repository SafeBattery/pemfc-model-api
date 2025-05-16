FROM apache/airflow:2.9.0

# PyTorch + CUDA 설치 (CUDA 11.8 예시)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 추가로 필요한 패키지 설치
RUN pip install pandas numpy scikit-learn torch
