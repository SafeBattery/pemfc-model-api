# /opt/airflow/dags/retrain_PWU_dag.py
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests
import os

def reload_model():
    model_path = "/models/PWU/model.pth"  # 절대경로로 고정
    data = {"model_path": model_path, "type": "PWU"}
    response = requests.post("http://flask-api:5000/reload_model", json=data)
    print(f"[retrain_PWU] Reload response → {response.text}")

# DAG 정의
with DAG(
    dag_id="retrain_PWU_dag",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    description="Retrain PWU model and notify Flask",
) as dag:

    train_model = BashOperator(
        task_id="train_PWU",
        bash_command="python /opt/airflow/scripts/train_PWU.py"
    )

    reload_model_task = PythonOperator(
        task_id="reload_PWU_model",
        python_callable=reload_model
    )

    train_model >> reload_model_task

dag = dag