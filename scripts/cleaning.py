import pandas as pd
from scripts.logging import LoggerConfig

class DataCleaner:
    def __init__(self, train, test):
        """Initialize train and test dataframe ."""
        self.train = train
        self.test = test
        LoggerConfig.log_message("Data cleaner objected created successfully.")

    def handle_missing_data(self):
        """Handle missing data in both train and test datasets."""
        self.train['Promo'].fillna(0, inplace=True)
        self.train['CompetitorDistance'].fillna(self.train['CompetitorDistance'].median(), inplace=True)
        
        self.test['Promo'].fillna(0, inplace=True)
        self.test['CompetitorDistance'].fillna(self.test['CompetitorDistance'].median(), inplace=True)
        
        LoggerConfig.log_message("Missing data handled.")
        return self

    def handle_outliers(self):
        """Handle outliers by capping extreme values in Sales."""
        q_low = self.train['Sales'].quantile(0.01)
        q_high = self.train['Sales'].quantile(0.99)
        self.train = self.train[(self.train['Sales'] > q_low) & (self.train['Sales'] < q_high)]
        LoggerConfig.log_message("Outliers handled in Sales.")
        return self

    def feature_engineering(self):
        """Add new features derived from existing data."""
        self.train['Date'] = pd.to_datetime(self.train['Date'])
        self.train['DayOfWeek'] = self.train['Date'].dt.dayofweek
        self.train['Month'] = self.train['Date'].dt.month
        self.train['Year'] = self.train['Date'].dt.year
        
        self.test['Date'] = pd.to_datetime(self.test['Date'])
        self.test['DayOfWeek'] = self.test['Date'].dt.dayofweek
        self.test['Month'] = self.test['Date'].dt.month
        self.test['Year'] = self.test['Date'].dt.year

        LoggerConfig.log_message("Feature engineering completed.")
        return self
