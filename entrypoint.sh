#!/bin/sh

# Set the PORT environment variable
export PORT=8080

# Run the MLflow UI
mlflow ui --backend-store-uri /mlflow --host 0.0.0.0 --port $PORT
