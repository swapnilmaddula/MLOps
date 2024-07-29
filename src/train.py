import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

iris_df = pd.read_csv("Data/dataset.csv")

X = iris_df.drop(columns=['target'])
y = iris_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_test.to_csv("Data/test_data.csv", index=False)

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

model_name = "sk-learn-random-forest-reg-model"

try:
    client.create_registered_model(model_name)
except mlflow.exceptions.RestException as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e):
        print(f"Model '{model_name}' already exists.")
    else:
        raise e

with mlflow.start_run() as run:
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    mlflow.log_param('n_estimators', 100)
    mlflow.log_param('random_state', 42)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.sklearn.log_model(clf, "model")

    client.create_model_version(
        name=model_name,
        source=f"runs:/{run.info.run_id}/model",
        run_id=run.info.run_id
    )

    print(f'Model version registered and logged in MLflow')

latest_model_uri = f"models:/{model_name}/latest"
loaded_model = mlflow.pyfunc.load_model(model_uri=latest_model_uri)

test_data = pd.read_csv("Data/test_data.csv")
predictions = loaded_model.predict(test_data)
print(predictions)
