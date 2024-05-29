from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from ucimlrepo import fetch_ucirepo
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)

def load_dataset():
    """
    Load the maternal health risk dataset and split it into training and test sets.

    Returns:
        tuple: A tuple containing training features (X_train), test features (X_test), 
               training target labels (y_train), and test target labels (y_test).
    """
    logging.info("Loading dataset...")
    maternal_health_risk = fetch_ucirepo(id=863)
    X = maternal_health_risk.data.features
    y = maternal_health_risk.data.targets

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info("Dataset loaded and split into train and test sets.")
    return X_train, X_test, y_train, y_test

def encode_labels(y):
    """
    Encode categorical target labels into numerical labels.

    Args:
        y (array-like): Target labels.

    Returns:
        array-like: Encoded target labels.
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return y_encoded

def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier model.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training target labels.

    Returns:
        RandomForestClassifier: Trained model.
    """
    logging.info("Training model...")
    # Define parameter grid for grid search
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10, 20]
    }
    
    # Initialize RandomForestClassifier
    rf = RandomForestClassifier()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

    # Flatten y_train values to a 1D array
    y_train = y_train.ravel()

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get best estimator
    best_estimator = grid_search.best_estimator_
    
    # Log the model with MLflow
    mlflow.sklearn.log_model(best_estimator, "model")
    logging.info("Model trained and logged to MLflow.")
    return best_estimator

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.

    Args:
        model (RandomForestClassifier): Trained model.
        X_test (array-like): Test features.
        y_test (array-like): Test target labels.
    """
    logging.info("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {accuracy}")

    # Log evaluation metrics using MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_param("max_depth", model.max_depth if model.max_depth is not None else -1)
    mlflow.log_param("min_samples_split", model.min_samples_split)

    # Log classification report
    report = classification_report(y_test, y_pred)
    logging.info(f"Classification Report:\n{report}")
    mlflow.log_text(report, "report.txt")

def main():
    try:
        # Set the MLflow tracking URI
        mlflow.set_tracking_uri("https://mlflow-ui-o33kkbr4pa-uc.a.run.app/")
        
        # Set the experiment name
        experiment_name = "mlops-bootcamp"
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run():
            X_train, X_test, y_train, y_test = load_dataset()
            y_train = encode_labels(y_train)
            y_test = encode_labels(y_test)
            model = train_model(X_train, y_train)
            evaluate_model(model, X_test, y_test)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        mlflow.log_artifact("error.log")
        raise

if __name__ == "__main__":
    main()
