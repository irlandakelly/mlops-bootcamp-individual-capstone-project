# run_eda.py

# Import necessary modules
import subprocess
import src.data.ingestion as ingestion
import src.exploratory_analysis.clean_data as clean_data
import src.exploratory_analysis.explore_data as explore_data
import src.exploratory_analysis.visualize_data as visualize_data

def run_eda(df):
    """
    Run the EDA functions on the provided DataFrame.

    Args:
        df (DataFrame): The dataset to analyze.
    """
    # Clean and explore data
    clean_data.explore_outliers(df)
    clean_data.identify_null_values(df)
    clean_data.create_histograms(df)
    explore_data.explore_data(df)
    explore_data.create_histograms(df)
    explore_data.explore_outliers(df)
    explore_data.explore_correlations(df)

    # Visualize data
    visualize_data.create_scatterplots(df)
    visualize_data.create_kde_plots(df)
    visualize_data.create_cdf_plots(df)

def main():
    """
    Main function to load data and execute EDA.
    """
    # Load the dataset
    X, y = ingestion.load_dataset()

    # Combine features and target into a single DataFrame
    df = X.copy()
    df['RiskLevel'] = y

    # Run EDA
    run_eda(df)

if __name__ == "__main__":
    main()