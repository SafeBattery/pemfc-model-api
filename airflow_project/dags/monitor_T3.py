from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
import requests
import numpy as np


# ✅ 모델 예측 및 retrain 필요 여부 판단
def monitor_model(**kwargs):
    # 입력 차원: 600 x 4 (P_H2_inlet, P_Air_inlet, T_Heater, T_Stack_inlet)
    X_test = np.tile(np.linspace(0.2, 0.8, 4), (600, 1)).tolist()
    y_true = [0.65]  # 임시 정답
    threshold = 0.05

    res = requests.post("http://flask-api:5000/predict", json={
        "input": X_test,
        "type": "T3",
        "threshold": threshold
    })

    if res.status_code != 200:
        raise Exception("Flask prediction failed.")

    prediction = res.json()["prediction"]
    error = abs(prediction - y_true[0])
    print(f"[monitor_T3] Prediction: {prediction}, Ground Truth: {y_true[0]}, Error: {error}")

    # 조건 만족 시 retrain 실행
    kwargs['ti'].xcom_push(key='retrain', value=(error > threshold))


# ✅ retrain 여부에 따라 브랜칭
def check_if_retrain_needed(**kwargs):
    retrain_flag = kwargs['ti'].xcom_pull(task_ids='monitor_model', key='retrain')
    return "trigger_retrain" if retrain_flag else "end"


# ✅ DAG 정의
with DAG(
        dag_id="monitor_T3",
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
        trigger_dag_id="retrain_T3_dag"
    )

    end = EmptyOperator(task_id="end")

    monitor >> check >> [trigger_retrain, end]

dag = dag
