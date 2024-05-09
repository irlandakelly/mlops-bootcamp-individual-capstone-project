from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def explore_outliers(df):
    """
    Explore outliers in numerical features using boxplots.

    Args:
    df (DataFrame): The dataset to explore.
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        sns.boxplot(x=df[column])
        plt.xlabel(column)
        plt.ylabel('Value')
        plt.title(f'Boxplot of {column}')
        plt.show()

def identify_null_values(df):
    """
    Identify null values in the dataset.

    Args:
    df (DataFrame): The dataset to inspect.
    """
    print("Null values per column:")
    print(df.isnull().sum())

def create_histograms(df):
    """
    Create histograms for each numerical feature.

    Args:
    df (DataFrame): The dataset to visualize.
    """
    for column in df.columns:
        plt.hist(df[column])
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {column}')
        plt.show()
