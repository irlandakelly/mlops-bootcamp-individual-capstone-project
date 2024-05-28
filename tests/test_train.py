import pytest
from sklearn.ensemble import RandomForestClassifier
from src.models.train import load_dataset, encode_labels, train_model, evaluate_model
import mlflow

def test_load_dataset():
    X_train, X_test, y_train, y_test = load_dataset()
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

def test_encode_labels():
    _, _, y_train, _ = load_dataset()
    y_encoded = encode_labels(y_train)
    assert y_encoded is not None
    assert len(y_encoded) == len(y_train)

def test_train_model():
    X_train, _, y_train, _ = load_dataset()
    y_train = encode_labels(y_train)
    with mlflow.start_run():
        model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    X_train, X_test, y_train, y_test = load_dataset()
    y_train = encode_labels(y_train)
    y_test = encode_labels(y_test)
    with mlflow.start_run():
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)

def test_end_to_end():
    X_train, X_test, y_train, y_test = load_dataset()
    y_train = encode_labels(y_train)
    y_test = encode_labels(y_test)
    with mlflow.start_run():
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators > 0
