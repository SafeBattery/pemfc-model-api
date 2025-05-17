from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
import requests
import numpy as np

# ✅ 모델 예측 및 retrain 필요 여부 판단
def monitor_model(**kwargs):
    X_test = np.tile(np.linspace(0.1, 0.9, 9), (600, 1)).tolist()
    y_true = [0.7, 3.5]
    threshold = 0.05

    res = requests.post("http://flask-api:5000/predict", json={
        "input": X_test,
        "type": "PWU",
        "threshold": threshold
    })

    if res.status_code != 200:
        print("[ERROR] Flask 응답 상태:", res.status_code)
        print("[ERROR] 응답 내용:", res.text)
        raise Exception("Flask prediction failed.")

    prediction = res.json()["prediction"]
    error = abs(prediction[0] - y_true[0])
    print(f"Prediction: {prediction[0]}, Ground Truth: {y_true[0]}, Error: {error}")
    kwargs['ti'].xcom_push(key='retrain', value=True)

# ✅ retrain 여부에 따라 브랜칭
def check_if_retrain_needed(**kwargs):
    retrain_flag = kwargs['ti'].xcom_pull(task_ids='monitor_model', key='retrain')
    return "trigger_retrain" if retrain_flag else "end"

# ✅ DAG 정의
with DAG(
    dag_id="monitor_PWU",
    start_date=days_ago(1),
    schedule_interval="@hourly",
    catchup=False,
    tags=["monitoring"]
) as dag:

    monitor = PythonOperator(
        task_id="monitor_model",
        python_callable=monitor_model
    )

    check = BranchPythonOperator(
        task_id="check_if_retrain_needed",
        python_callable=check_if_retrain_needed
    )

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="retrain_PWU_dag"
    )

    end = EmptyOperator(task_id="end")

    monitor >> check >> [trigger_retrain, end]
dag = dag
