import mlflow.sklearn

# Load the model
model_uri = "sk-learn-random-forest-reg-model"
loaded_model = mlflow.sklearn.load_model(model_uri)

# Make predictions
predictions = loaded_model.predict(X_test)
