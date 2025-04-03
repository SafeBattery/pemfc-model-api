#!/bin/bash

echo "[INFO] Running install_dependencies.sh"

# Docker 설치 여부 확인 및 설치
if ! command -v docker &> /dev/null
then
    echo "[INFO] Docker not found. Installing..."
    sudo apt update
    sudo apt install -y docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
else
    echo "[INFO] Docker is already installed."
fi

# 권한 설정
sudo chmod -R 755 /home/ubuntu/flask_test
