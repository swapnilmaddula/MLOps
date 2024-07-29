from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 7, 22),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# Define the DAG
main_pipeline = DAG(
    'main_pipeline',
    default_args=default_args,
    description='A simple ML pipeline DAG',
    schedule_interval=timedelta(days=1),
)

# Task to load data
load_data_task = BashOperator(
    task_id='load_data',
    bash_command='python3 src/source_to_lake.py',
    dag=main_pipeline,
)

# Task to train the model
train_model_task = BashOperator(
    task_id='train_model',
    bash_command='python3 src/train.py',
    dag=main_pipeline,
)

# Task to perform inference
infer_model_task = BashOperator(
    task_id='infer_model',
    bash_command='python3 src/inference.py',
    dag=main_pipeline,
)

# Set up task dependencies
load_data_task >> train_model_task >> infer_model_task
