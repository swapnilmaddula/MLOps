#!/bin/sh
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
mlflow db upgrade sqlite:///mlflow.db 

python src/source_to_lake.py
python src/train.py
python src/inference.py