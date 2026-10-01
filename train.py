import subprocess
import pandas as pd
import mlflow
import mlflow.sklearn
import subprocess

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
data = pd.read_csv("data.csv")

X = data[["area", "bedrooms"]]
y = data["price"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Define model
model = LinearRegression(fit_intercept=True)


# Set MLflow experiment
mlflow.set_experiment("House_Price_Prediction")

with mlflow.start_run():

    dataset_version = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", "data.csv.dvc"]
    ).decode().strip()

    mlflow.log_param("dataset_version", dataset_version)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    # Log parameters
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_param("fit_intercept", True)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # Log model
    mlflow.sklearn.log_model(
        model,
       name =  "house_price_model"
    )

    print("Model trained successfully!")
    print(f"MAE  : {mae:.2f}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")
