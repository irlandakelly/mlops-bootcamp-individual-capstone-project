from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from ucimlrepo import fetch_ucirepo
import mlflow

def load_dataset():
    """
    Load the maternal health risk dataset.

    Returns:
        tuple: A tuple containing features (X) and target labels (y).
    """
    maternal_health_risk = fetch_ucirepo(id=863)
    X = maternal_health_risk.data.features
    y = maternal_health_risk.data.targets
    return X, y

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

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get best estimator
    best_estimator = grid_search.best_estimator_

    return best_estimator

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.

    Args:
        model (RandomForestClassifier): Trained model.
        X_test (array-like): Test features.
        y_test (array-like): Test target labels.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Log evaluation metrics using MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_param("max_depth", model.max_depth)
    mlflow.log_param("min_samples_split", model.min_samples_split)

    # Log classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)
    mlflow.log_text(report, "report.txt")

def main():
    """
    Main function to load data, train model, and evaluate performance.
    """
    # Load the dataset
    X, y = load_dataset()

    # Encode target labels
    y_encoded = encode_labels(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train the model
    with mlflow.start_run():
        model = train_model(X_train, y_train)

        # Evaluate the model
        evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
