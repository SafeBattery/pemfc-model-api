# /opt/airflow/dags/retrain_T3_dag.py
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests
import os

def reload_model():
    model_path = "/models/T3/model.pth"  # 절대 경로
    data = {"model_path": model_path, "type": "T3"}
    response = requests.post("http://flask-api:5000/reload_model", json=data)
    print(f"[retrain_T3] Reload response → {response.text}")

# DAG 정의
with DAG(
    dag_id="retrain_T3_dag",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    description="Retrain T3 model and notify Flask",
) as dag:

    train_model = BashOperator(
        task_id="train_T3",
        bash_command="python /opt/airflow/scripts/train_T3.py"
    )

    reload_model_task = PythonOperator(
        task_id="reload_T3_model",
        python_callable=reload_model
    )

    train_model >> reload_model_task

dag = dag
