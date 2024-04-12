# Wizeline MLOps Bootcamp Individual Capstone Project

The Wizeline MLOps Bootcamp Individual Capstone Project focuses on implementing end-to-end machine learning operations (MLOps) practices in a real-world scenario. Utilizing Python and popular machine learning libraries, this project aims to develop and deploy a predictive model for assessing maternal health risks. The dataset, sourced from real-world healthcare settings, encompasses vital health indicators such as age, blood pressure, blood sugar levels, body temperature, and heart rate. By leveraging advanced data analysis, model training, and deployment strategies, this project endeavors to create a robust solution for predicting maternal health risks, thereby contributing to improved healthcare outcomes and supporting decision-making processes. Through rigorous exploration, modeling, and deployment phases, this project serves as a comprehensive demonstration of MLOps principles and techniques applied to a critical healthcare domain.

## Setup

1. **Clone the repository:**

```
git clone https://github.com/irlandakelly/mlops-bootcamp-individual-capstone-project.git
```


2. **Navigate to the project directory:**

```
cd mlops-bootcamp-individual-capstone-project
```


3. **Create and activate a virtual environment:**
- If you're using `venv`, run:
  ```
  python -m venv env
  ```
  - **Windows:**
    ```
    .\env\Scripts\activate
    ```
  - **Unix/MacOS:**
    ```
    source env/bin/activate
    ```

4. **Install dependencies:**

```
pip install -r requirements.txt
```


## Directory Structure

├── data<br>
│   ├── external<br>
│   ├── interim<br>
│   ├── processed<br>
│   ├── raw<br>
│   │   ├── data.csv<br>
├── models<br>
├── notebooks<br>
│   └── Maternal_Health_Risk_Data_EDA.ipynb<br>
├── src<br>
│   ├── data<br>
│   │   └── ingestion.py<br>
│   ├── models<br>
│   │   └── train.py<br>
│   └── visualization<br>
├── requirements.txt<br>
└── README.md<br>



## Description of `ingestion.py`

The `ingestion.py` file is responsible for importing and preparing the dataset for use in the project. It retrieves the maternal health risk dataset from the UCI ML Repository, loads it into pandas dataframes, and provides metadata and variable information.

## Description of `train.py`

The `train.py` file contains functions to train and evaluate a machine learning model using the fetched dataset. It splits the data into training and testing sets, encodes categorical labels, trains a RandomForestClassifier model, and evaluates its performance.

## Instructions for Running `ingestion.py` and `train.py`

To run `ingestion.py`, execute the following command:

```
python src/data/ingestion.py
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
- **Model Accuracy:** The trained Random Forest Classifier model achieved an accuracy of **0.8275862068965517** on the test dataset. 
