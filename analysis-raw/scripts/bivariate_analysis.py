from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Abstract Base Class for Bivariate Analysis Strategy
class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Perform bivariate analysis on two features of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first feature/column to be analyzed.
        feature2 (str): The name of the second feature/column to be analyzed.

        Returns:
        None: This method visualizes the relationship between the two features.
        """
        pass


# Concrete Strategy for Numerical vs Numerical Analysis
# ------------------------------------------------------
# This strategy analyzes the relationship between two numerical features using scatter plots.
class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between two numerical features using a scatter plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first numerical feature/column to be analyzed.
        feature2 (str): The name of the second numerical feature/column to be analyzed.

        Returns:
        None: Displays a scatter plot showing the relationship between the two features.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()


# Concrete Strategy for Categorical vs Numerical Analysis
# --------------------------------------------------------
# This strategy analyzes the relationship between a categorical feature and a numerical feature using box plots.
class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between a categorical feature and a numerical feature using a box plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the categorical feature/column to be analyzed.
        feature2 (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays a box plot showing the relationship between the categorical and numerical features.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()

# Concrete Strategy for Categorical vs Categorical Analysis
# --------------------------------------------------------
# This strategy analyzes the relationship between two categorical features using stacked bar plots.
class CategoricalVsCategoricalAnalysis:
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between two categorical features using a 100% stacked bar chart with data labels.
        The chart is only generated if the number of categories in each feature is less than 10.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first categorical feature/column to be analyzed.
        feature2 (str): The name of the second categorical feature/column to be analyzed.

        Returns:
        None: Displays a 100% stacked bar chart if the number of categories is less than 10,
              otherwise displays an error message.
        """
        # Check the number of unique categories in each feature
        num_categories_feature1 = df[feature1].nunique()
        num_categories_feature2 = df[feature2].nunique()

        # If either feature has 10 or more categories, display an error and return
        if num_categories_feature1 > 10 or num_categories_feature2 > 10:
            print(f"Error: The number of categories in '{feature1}' is {num_categories_feature1} "
                  f"and in '{feature2}' is {num_categories_feature2}. "
                  "The chart will not be generated for features with more than 10 categories.")
            return

        # Create a cross-tabulation of the two categorical features
        cross_tab = pd.crosstab(df[feature1], df[feature2], normalize='index') * 100

        # Plot the 100% stacked bar chart
        ax = cross_tab.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')

        # Add title and labels
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel("Percentage")
        plt.xticks(rotation=45)

        # Add data labels
        for container in ax.containers:
            # Get the x and y positions for each segment
            for bar in container:
                height = bar.get_height()
                width = bar.get_width()
                x = bar.get_x() + width / 2
                y = bar.get_y() + height / 2

                # Only add labels for segments with height > 0 to avoid clutter
                if height > 0:
                    ax.text(
                        x, y,
                        f"{height:.1f}%",  # Format the label to 1 decimal place
                        ha='center', va='center',
                        fontsize=8,  # Small font size
                        color='black'
                    )

        # Display the legend
        plt.legend(title=feature2, bbox_to_anchor=(1.05, 1), loc='upper left')

        # Show the plot
        plt.tight_layout()
        plt.show()

# Context Class that uses a BivariateAnalysisStrategy
# ---------------------------------------------------
# This class allows for switching between different bivariate analysis strategies.
class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        """
        Initializes the BivariateAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (BivariateAnalysisStrategy): The strategy to be used for bivariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        """
        Sets a new strategy for the BivariateAnalyzer.

        Parameters:
        strategy (BivariateAnalysisStrategy): The new strategy to be used for bivariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Executes the bivariate analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first feature/column to be analyzed.
        feature2 (str): The name of the second feature/column to be analyzed.

        Returns:
        None: Executes the strategy's analysis method and visualizes the results.
        """
        self._strategy.analyze(df, feature1, feature2)