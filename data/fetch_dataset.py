# Install the ucimlrepo package
# !pip install ucimlrepo

# Import necessary libraries
from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch dataset
maternal_health_risk = fetch_ucirepo(id=863)

# Data (as pandas dataframes)
X = maternal_health_risk.data.features
y = maternal_health_risk.data.targets

# Metadata
metadata = maternal_health_risk.metadata

# Variable information
variables = maternal_health_risk.variables

# View the full documentation
print("Dataset Metadata:")
print(metadata)
print("\nVariable Information:")
print(variables)
