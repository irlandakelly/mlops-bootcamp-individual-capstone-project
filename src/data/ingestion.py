# ingestion.py

# Import necessary libraries
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_dataset():
    # Fetch dataset
    maternal_health_risk = fetch_ucirepo(id=863)

    # Data (as pandas dataframes)
    X = maternal_health_risk.data.features
    y = maternal_health_risk.data.targets

    # Binning
    X['Age_bin'] = pd.cut(X['Age'], bins=[20, 30, 40, 50, 60, 70, 80], labels=False)

    # Interaction Features
    X['BP_Ratio'] = X['SystolicBP'] / X['DiastolicBP']

    # Polynomial Features
    X['Age_squared'] = X['Age'] ** 2
    X['HeartRate_cubed'] = X['HeartRate'] ** 3

    # Outlier Treatment
    for column in ['BS', 'BodyTemp', 'HeartRate']:
        Q1 = X[column].quantile(0.25)
        Q3 = X[column].quantile(0.75)
        IQR = Q3 - Q1
        X[column] = X[column].clip(lower=Q1-1.5*IQR, upper=Q3+1.5*IQR)

    # One-Hot Encoding
    encoder = OneHotEncoder(sparse=False)
    risk_level_encoded = encoder.fit_transform(X[['RiskLevel']])
    X = pd.concat([X, pd.DataFrame(risk_level_encoded, columns=encoder.get_feature_names(['RiskLevel']))], axis=1)
    X.drop('RiskLevel', axis=1, inplace=True)

    # Normalization
    scaler = StandardScaler()
    X[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']] = scaler.fit_transform(X[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']])

    return X, y