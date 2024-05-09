# Wizeline MLOps Bootcamp Individual Capstone Project

The Wizeline MLOps Bootcamp Individual Capstone Project is centered on the application of end-to-end machine learning operations (MLOps) practices in a practical context. The project employs Python and widely-used machine learning libraries to build and deploy a predictive model for evaluating maternal health risks.

The dataset, derived from real-world healthcare environments, includes crucial health indicators such as age, blood pressure, blood sugar levels, body temperature, and heart rate. The project leverages sophisticated data analysis, model training, and deployment strategies to create a robust solution for predicting maternal health risks. This contributes to enhancing healthcare outcomes and aids in decision-making processes.

The project involves rigorous exploration, modeling, and deployment phases, serving as a comprehensive demonstration of MLOps principles and techniques in a vital healthcare domain. The project includes data ingestion and preprocessing, model training and hyperparameter tuning, model evaluation, and tracking of model metrics and parameters using MLflow. The project also demonstrates best practices in version control and reproducibility in machine learning workflows.

## Setup

1. **Clone the repository:**

```
git clone https://github.com/irlandakelly/mlops-bootcamp-individual-capstone-project.git
```


2. **Navigate to the project directory:**

```
cd mlops-bootcamp-individual-capstone-project
```


3. **Build the Docker image:**
  ```
  docker build -t mlops-bootcamp .
  ```
  

4. **Create a Docker volume for MLflow data:**

```
docker volume create mlflow_data
```


5. **Run the Docker container with the volume mounted:**

```
docker run -v mlflow_data:/mlflow -p 5000:5000 mlops-bootcamp
```

After running the Docker container, you can view the MLflow experiment tracking UI by navigating to `http://localhost:5000` in your web browser. The MLflow data will be stored in the `mlflow_data` Docker volume and will persist even if the container is stopped or deleted.

## Directory Structure

├── notebooks<br>
│   └── Maternal_Health_Risk_Data_EDA.ipynb<br>
├── src<br>
│   ├── data<br>
│   │   └── ingestion.py<br>
│   ├── exploratory_analysis<br>
│   │   ├── clean_data.py<br>
│   │   ├── explore_data.py<br>
│   │   └── visualize_data.py<br>
│   ├── models<br>
│   │   └── train.py<br>
│   └── visualization<br>
├── Dockerfile.txt<br>
├── README.md<br>
├── requirements.txt<br>
└── run_eda.py<br>



## Description of `ingestion.py`

The `ingestion.py` script is responsible for data ingestion and preprocessing. It fetches the maternal health risk dataset from the UCI ML Repository using the `fetch_ucirepo` function. The data is then loaded into pandas dataframes `X` (features) and `y` (targets).

The script performs several preprocessing steps:

Binning: The 'Age' feature is binned into categories using pandas' cut function.

Interaction Features: A new feature 'BP_Ratio' is created as the ratio of 'SystolicBP' to 'DiastolicBP'.

Polynomial Features: Two new polynomial features 'Age_squared' and 'HeartRate_cubed' are created by squaring the 'Age' feature and cubing the 'HeartRate' feature, respectively.

Outlier Treatment: Outliers in the 'BS', 'BodyTemp', and 'HeartRate' features are handled by clipping values outside 1.5 times the Interquartile Range (IQR) above the third quartile (Q3) or below the first quartile (Q1).

One-Hot Encoding: The 'RiskLevel' feature is one-hot encoded using the `OneHotEncoder` from sklearn, resulting in binary (0/1) columns for each risk level. The original 'RiskLevel' column is then dropped.

Normalization: The 'Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', and 'HeartRate' features are normalized (scaled to have zero mean and unit variance) using the `StandardScaler` from sklearn.

The script returns the preprocessed features `X` and targets `y`.

## Description of `run_eda.py`
The `run_eda.py` script is responsible for performing exploratory data analysis (EDA) on the provided dataset. It includes the following functions:

run_eda: This function executes a series of EDA tasks on the provided DataFrame. It first cleans the data by exploring outliers and identifying null values. Then, it explores the data by generating histograms, exploring outliers again, and analyzing correlations. Finally, it visualizes the data by creating scatterplots, kernel density estimation (KDE) plots, and cumulative distribution function (CDF) plots.

main: The `main` function serves as the entry point for the script. It loads the dataset using the `load_dataset` function from the ingestion module. It combines the features and target labels into a single DataFrame and calls the `run_eda` function to perform EDA on the dataset.

The script is designed to be executed as a standalone program. When run, it loads the dataset, conducts exploratory analysis, and visualizes the findings.

## Description of `train.py`

The `train.py` script is responsible for training and evaluating a RandomForestClassifier model on the maternal health risk dataset. It includes the following functions:

load_dataset: Fetches the maternal health risk dataset from the UCI ML Repository and returns the features (X) and target labels (y).

encode_labels: Encodes categorical target labels into numerical labels using sklearn's `LabelEncoder`.

train_model: Trains a RandomForestClassifier model using GridSearchCV for hyperparameter tuning. The best model is returned.

evaluate_model: Evaluates the trained model on the test set, prints the accuracy and classification report, and logs these metrics and the model parameters using MLflow.

The `main` function orchestrates the process: it loads the dataset, encodes the target labels, splits the data into training and testing sets, trains the model, and evaluates it. The training and evaluation process is wrapped in an MLflow run for experiment tracking.

The script is intended to be run as a standalone program. When run, it calls the `main` function.

## Instructions for Running `ingestion.py`, `run_eda.py` and `train.py`

To run `ingestion.py`, execute the following command:

```
python src/data/ingestion.py
```

To run `ingestion.py`, execute the following command:

```
python run_eda.py
```

To run `train.py`, execute the following command:

```
python src/models/train.py
```

## Dataset Description

- **UCI ID:** 863
- **Name:** Maternal Health Risk
- **Repository URL:** [Maternal Health Risk Dataset](https://archive.ics.uci.edu/dataset/863/maternal+health+risk)
- **Data URL:** [Data CSV](https://archive.ics.uci.edu/static/public/863/data.csv)
- **Abstract:** Data has been collected from different hospitals, community clinics, maternal health cares from the rural areas of Bangladesh through the IoT based risk monitoring system.
- **Area:** Health and Medicine
- **Tasks:** Classification
- **Characteristics:** Multivariate
- **Number of Instances:** 1013
- **Number of Features:** 6
- **Feature Types:** Real, Integer
- **Demographics:** Age
- **Target Column:** RiskLevel
- **Has Missing Values:** No
- **Year of Dataset Creation:** 2020
- **Last Updated:** Fri Nov 03 2023
- **Dataset DOI:** [10.24432/C5DP5D](https://doi.org/10.24432/C5DP5D)
- **Creators:** Marzia Ahmed
- **Introduction Paper:**
  - Title: Review and Analysis of Risk Factor of Maternal Health in Remote Area Using the Internet of Things (IoT)
  - Authors: Marzia Ahmed, M. A. Kashem, Mostafijur Rahman, S. Khatun
  - Published In: Lecture Notes in Electrical Engineering, vol 632
  - Year: 2020
  - URL: [Paper URL](https://www.semanticscholar.org/paper/f175092a3b2217c9abca5bf5d91bab3c245c6b10)
- **Additional Information:**
  - **Summary:** Age, Systolic Blood Pressure as SystolicBP, Diastolic BP as DiastolicBP, Blood Sugar as BS, Body Temperature as BodyTemp, HeartRate and RiskLevel. All these are the responsible and significant risk factors for maternal mortality, that is one of the main concerns of SDG of UN.
- **Model Accuracy:** The trained Random Forest Classifier model achieved an accuracy of **0.8078817733990148** on the test dataset. 
