import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn


def train_model(alpha, l1_ratio):
    # Generate some sample data
    X = np.random.rand(100, 1)
    y = 2 * X + np.random.randn(100, 1)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    return mse, model


if __name__ == "__main__":
    alpha = 0.03
    l1_ratio = 0.03

    # Start MLflow run
    with mlflow.start_run():
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        mse, model = train_model(alpha, l1_ratio)
        mlflow.log_metric("mse", mse)

        # Log the model
        mlflow.sklearn.log_model(model, "model")