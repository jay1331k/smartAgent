import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_acquisition_and_preprocessing import get_data  # Import from previous script
from data_integration_and_analysis import integrate_and_analyze

def visualize_data(merged_data):
    """
    Creates visualizations to explore the relationship between vaccination rates and case fatality rates.

    Args:
        merged_data (pd.DataFrame): The integrated dataframe.

    Returns:
        None.  Displays plots.
    """

    if merged_data is None:
        print("No data to visualize.")
        return

    try:
        # Scatter plot: Vaccination Rate vs. Case Fatality Rate
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='fully_vaccinated_per_hundred', y='case_fatality_rate', data=merged_data)
        plt.title('Vaccination Rate vs. Case Fatality Rate')
        plt.xlabel('Fully Vaccinated per Hundred')
        plt.ylabel('Case Fatality Rate')
        plt.show()

        # Time series plot (example for a specific country)
        if not merged_data.empty:  # Check if merged_data is not empty
            country_to_plot = merged_data['country'].iloc[0]  # Get the first country
            country_data = merged_data[merged_data['country'] == country_to_plot]

            plt.figure(figsize=(12, 6))
            plt.plot(country_data['date'], country_data['fully_vaccinated_per_hundred'], label='Fully Vaccinated per Hundred')
            plt.plot(country_data['date'], country_data['case_fatality_rate'], label='Case Fatality Rate')
            plt.title(f'Time Series for {country_to_plot}')
            plt.xlabel('Date')
            plt.ylabel('Rate')
            plt.legend()
            plt.show()
        else:
            print("Merged data is empty. Cannot create time series plot.")

        #  Distribution plot of vaccination rates
        plt.figure(figsize=(10, 6))
        sns.histplot(merged_data['fully_vaccinated_per_hundred'], kde=True)
        plt.title('Distribution of Vaccination Rates')
        plt.xlabel('Fully Vaccinated per Hundred')
        plt.show()

        # Distribution plot of case fatality rates.
        plt.figure(figsize=(10, 6))
        sns.histplot(merged_data['case_fatality_rate'], kde=True)
        plt.title('Distribution of Case Fatality Rates')
        plt.xlabel('Case Fatality Rate')
        plt.show()

    except Exception as e:
        print(f"Error during data visualization: {e}")


if __name__ == '__main__':
    # Example Usage (replace with your actual file paths or URLs)
    vaccination_url = 'vaccination_data.csv'  # Same as in previous script
    case_fatality_url = 'case_fatality_data.csv'  # Same as in previous script

    vaccination_data, case_fatality_data = get_data(vaccination_url, case_fatality_url)

    if vaccination_data is not None and case_fatality_data is not None:
        merged_data = integrate_and_analyze(vaccination_data, case_fatality_data)
        visualize_data(merged_data)
    else:
        print("Data visualization failed.")