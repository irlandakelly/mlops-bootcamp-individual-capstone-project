import os
import seaborn as sns
import matplotlib.pyplot as plt

# Create a directory for the plots if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

def create_scatterplots(df):
    """
    Create scatterplots for each numerical feature against the target variable.

    Args:
    df (DataFrame): The dataset to visualize.
    """
    for i, column in enumerate(df.columns[:-1]):  # Exclude the target variable
        sns.scatterplot(x=column, y='RiskLevel', data=df)
        plt.xlabel(column)
        plt.ylabel('RiskLevel')
        plt.title(f'Scatterplot of {column} vs. RiskLevel')
        image_path = f'plots/scatterplot_{i}.png'
        plt.savefig(image_path)  # Save the figure to a file
        os.system(f'dvc add {image_path}')  # Add the image to DVC
        plt.clf()  # Clear the figure for the next plot

def create_kde_plots(df):
    """
    Create Kernel Density Estimation (KDE) plots for each numerical feature.

    Args:
    df (DataFrame): The dataset to visualize.
    """
    for i, column in enumerate(df.columns[:-1]):  # Exclude the target variable
        sns.kdeplot(df[column], fill=True)
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.title(f'Kernel Density Estimation (KDE) Plot of {column}')
        image_path = f'plots/kde_plot_{i}.png'
        plt.savefig(image_path)  # Save the figure to a file
        os.system(f'dvc add {image_path}')  # Add the image to DVC
        plt.clf()  # Clear the figure for the next plot

def create_cdf_plots(df):
    """
    Create Cumulative Distribution Function (CDF) plots for each numerical feature.

    Args:
    df (DataFrame): The dataset to visualize.
    """
    for i, column in enumerate(df.columns[:-1]):  # Exclude the target variable
        sns.kdeplot(df[column], cumulative=True)
        plt.xlabel(column)
        plt.ylabel('Cumulative Probability')
        plt.title(f'Cumulative Distribution Function (CDF) Plot of {column}')
        image_path = f'plots/cdf_plot_{i}.png'
        plt.savefig(image_path)  # Save the figure to a file
        os.system(f'dvc add {image_path}')  # Add the image to DVC
        plt.clf()  # Clear the figure for the next plot