import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def explore_data(df):
    """
    Explore the dataset by displaying the first few rows, summary statistics,
    and unique values of each column.

    Args:
    df (DataFrame): The dataset to explore.
    """
    # Display the first few rows of the dataset
    print("First few rows of the dataset:")
    print(df.head())

    # Get the summary statistics of the dataset
    print("\nSummary statistics of the dataset:")
    print(df.describe())

    # Count the number of unique values in each column
    print("\nNumber of unique values in each column:")
    print(df.nunique())

def create_histograms(df):
    """
    Create histograms for each numerical column in the dataset.

    Args:
    df (DataFrame): The dataset to visualize.
    """
    for column in df.columns:
        if df[column].dtype in [np.int64, np.float64]:
            plt.hist(df[column])
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.title(f'Histogram of {column}')
            plt.show()

def explore_outliers(df):
    """
    Explore outliers in the dataset using boxplots.

    Args:
    df (DataFrame): The dataset to visualize.
    """
    for column in df.columns:
        if df[column].dtype in [np.int64, np.float64]:
            sns.boxplot(x=df[column])
            plt.xlabel(column)
            plt.ylabel('Value')
            plt.title(f'Boxplot of {column}')
            plt.show()

            # Calculate the z-scores for the column
            z_scores = (df[column] - df[column].mean()) / df[column].std()

            # Define the threshold for considering an observation as an outlier
            threshold = 3

            # Identify the outliers
            outliers = df[abs(z_scores) > threshold]

            # Print the outliers
            print(f"Outliers in {column}:")
            print(outliers)
            print()

def explore_correlations(df):
    """
    Explore correlations between numerical features in the dataset.

    Args:
    df (DataFrame): The dataset to visualize.
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()