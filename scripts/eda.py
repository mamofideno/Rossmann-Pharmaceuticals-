
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from scripts.logging import LoggerConfig

class EDA:
    def __init__(self, train_df,test_df, promotion_col: str) -> None:
        self.train_df=train_df
        self.test_df=test_df
        self.promotion_col = promotion_col
    def get_promotion_distribution(self):
        train_promotion_dist = self.train_df[self.promotion_col].value_counts(normalize=True) * 100
        test_promotion_dist = self.test_df[self.promotion_col].value_counts(normalize=True) * 100
        return train_promotion_dist, test_promotion_dist
    def plot_promotion_distribution(self, train_promotion_dist, test_promotion_dist):
        """
        Plots the distribution of promotions for visual comparison between training and test sets.
        Args:
        - train_promotion_dist (pd.Series): Distribution of promotions in the training set.
        - test_promotion_dist (pd.Series): Distribution of promotions in the test set.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        sns.barplot(x=train_promotion_dist.index, y=train_promotion_dist.values, ax=ax[0], palette="Blues")
        ax[0].set_title("Promotion Distribution in Training Set")
        ax[0].set_xlabel("Promotion")
        ax[0].set_ylabel("Percentage")
        
        sns.barplot(x=test_promotion_dist.index, y=test_promotion_dist.values, ax=ax[1], palette="Greens")
        ax[1].set_title("Promotion Distribution in Test Set")
        ax[1].set_xlabel("Promotion")
        ax[1].set_ylabel("Percentage")
        
        plt.tight_layout()
        plt.show()
    
    def chi_square_test(self):
        """
        Performs a Chi-Square test to compare the distribution of promotions between training and test datasets.
        Prints the Chi-Square statistic and p-value, and interprets the result.
        """
        # Creating contingency table
        contingency_table = pd.crosstab(self.train_df[self.promotion_col], self.test_df[self.promotion_col], margins=False)
        
        # Perform Chi-Square test
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        
        print(f"Chi-Square Test Statistic: {chi2}")
        print(f"P-value: {p}")
        
        if p < 0.05:
            print("There is a significant difference in promotion distribution between training and test sets.")
        else:
            print("The promotion distribution is similar between training and test sets.")
    
    def analyze_promotion_distribution(self):
        """
        High-level method to perform the full analysis on the promotion distribution.
        1. Compute distribution
        2. Plot distribution
        3. Perform Chi-Square test
        """
        # Step 1: Get promotion distribution
        train_promotion_dist, test_promotion_dist = self.get_promotion_distribution()
        
        # Print distributions
        print("Training Promotion Distribution (%)\n", train_promotion_dist)
        print("Test Promotion Distribution (%)\n", test_promotion_dist)
        
        # Step 2: Plot distribution
        self.plot_promotion_distribution(train_promotion_dist, test_promotion_dist)
        
        # Step 3: Perform Chi-Square test
        self.chi_square_test()
    def compute_correlation(self):
        """Compute the correlation between Sales and Customers."""
        correlation = self.train_df['Sales'].corr(self.train_df['Customers'])
        LoggerConfig.log_message(f"Correlation between Sales and Customers: {correlation}")
        return correlation

    def analyze_promo_effect(self):
        """Analyze the effect of promotions on Sales and Customers."""
        promo_sales = self.train_df.groupby('Promo')['Sales'].mean()
        promo_customers = self.train_df.groupby('Promo')['Customers'].mean()
        LoggerConfig.log_message("Promo effect on sales and customers analyzed.")
        return promo_sales, promo_customers

    def assortment_sales(self):
        """Analyze sales based on assortment type."""
        assortment_sales = self.train_df.groupby('Assortment')['Sales'].mean()
        LoggerConfig.log_message("Assortment type effect on sales analyzed.")
        return assortment_sales

    def competitor_distance_sales_correlation(self):
        """Analyze the correlation between competitor distance and sales."""
        correlation = self.train_df['CompetitionDistance'].corr(self.train_df['Sales'])
        LoggerConfig.log_message(f"Correlation between Competitor Distance and Sales: {correlation}")
        return correlation    

