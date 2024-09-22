import seaborn as sns
import matplotlib.pyplot as plt
from scripts.logging import LoggerConfig

class DataVisualizer:
    def __init__(self, train, test):
        """Initialize with cleaned train and test data."""
        self.train = train
        self.test = test

    def promo_distribution(self):
        """Compare Promo distribution between training and test sets."""
        plt.figure(figsize=(10, 6))
        sns.histplot(self.train['Promo'], color='blue', label='Training', kde=True)
        sns.histplot(self.test['Promo'], color='red', label='Test', kde=True)
        plt.title('Promo Distribution - Train vs Test')
        plt.legend()
        plt.show()
        LoggerConfig.log_message("Promo distribution plotted for train and test.")

    def sales_before_during_after_holidays(self):
        """Plot sales behavior around holidays."""
        self.train['StateHoliday'] = self.train['StateHoliday'].replace(0, 'No Holiday')
        holiday_sales = self.train.groupby('StateHoliday')['Sales'].mean()
        holiday_sales.plot(kind='bar', title='Sales Before, During, and After Holidays')
        plt.show()
        LoggerConfig.log_message("Sales behavior around holidays visualized.")

    def seasonal_sales_trend(self):
        """Visualize seasonal sales trends."""
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=self.train, x='Month', y='Sales', hue='Year', ci=None)
        plt.title("Seasonal Sales Trends")
        plt.show()
        LoggerConfig.log_message("Seasonal sales trends visualized.")

    def correlation_sales_customers(self):
        """Plot the correlation between Sales and Customers."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Customers', y='Sales', data=self.train)
        plt.title("Sales vs Customers Correlation")
        plt.show()
        LoggerConfig.log_message("Sales vs Customers correlation visualized.")
