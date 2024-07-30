#!/bin/sh

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000 &

sleep 5

python3 src/source_to_lake.py
python3 src/train.py
python3 src/inference.py
