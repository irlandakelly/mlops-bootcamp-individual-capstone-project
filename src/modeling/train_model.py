from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo

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
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

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
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
