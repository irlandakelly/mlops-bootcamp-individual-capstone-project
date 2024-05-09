# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install DVC
RUN pip install dvc

# Initialize DVC
RUN dvc init --no-scm

# Set the MLflow tracking URI to the local file system
ENV MLFLOW_TRACKING_URI=/mlflow
RUN mkdir /mlflow

# Expose port 5000 for MLflow UI
EXPOSE 5000

# Start MLflow UI and Gunicorn
CMD ["mlflow", "ui", "--backend-store-uri", "/mlflow", "--host", "0.0.0.0"]
