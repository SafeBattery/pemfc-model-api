#!/bin/bash

echo "[INFO] Running start_server.sh"

cd /home/ubuntu/flask_test

# 기존 컨테이너 종료 및 삭제
sudo docker stop pw-api || true
sudo docker rm pw-api || true

# 새로 빌드하고 실행
sudo docker build -t pw-api .
sudo docker run -d --name pw-api -p 5000:5000 pw-api
