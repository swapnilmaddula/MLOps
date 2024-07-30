import mlflow.sklearn
import pandas as pd
# Load the model
mlflow.set_tracking_uri("http://127.0.0.1:5000")
model_name = "sk-learn-random-forest-reg-model"
latest_model_uri = f"models:/{model_name}/latest"
loaded_model = mlflow.pyfunc.load_model(model_uri=latest_model_uri)

X_test = pd.read_csv("Data/test_data.csv")
# Make predictions
predictions = loaded_model.predict(X_test)

print(predictions)
