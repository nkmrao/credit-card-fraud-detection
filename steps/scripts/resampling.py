from abc import ABC, abstractmethod

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Abstract Base Class for Resampling Strategy
class ResamplingStrategy(ABC):
    @abstractmethod
    def resample(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Abstract method to resample the dataset.

        Parameters:
        df (pd.DataFrame): The input DataFrame with the target variable.
        target_column (str): The name of the target column.

        Returns:
        pd.DataFrame: The resampled DataFrame.
        """
        pass

# Concrete Strategy for Oversampling the Minority Class
class OversamplingStrategy(ResamplingStrategy):
    def __init__(self):
        pass

    def resample(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Oversamples the minority class to balance the classes.

        Parameters:
        df (pd.DataFrame): The input DataFrame with the target variable.
        target_column (str): The name of the target column.

        Returns:
        pd.DataFrame: The oversampled DataFrame.
        """
        # Get class distribution
        class_counts = df[target_column].value_counts()
        max_count = class_counts.max()

        # Oversample each class
        df_resampled = pd.concat([
            df[df[target_column] == cls].sample(n=max_count, replace=True, random_state=42)
            for cls in class_counts.index
        ], ignore_index=True)

        return df_resampled
    
# Concrete Strategy for Undersampling the Majority Class
class UndersamplingStrategy(ResamplingStrategy):
    def __init__(self):
        pass

    def resample(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Undersamples the majority class to balance the dataset.

        Parameters:
        df (pd.DataFrame): The input DataFrame with the target variable.
        target_column (str): The name of the target column.

        Returns:
        pd.DataFrame: The undersampled DataFrame.
        """
        # Get class distribution
        class_counts = df[target_column].value_counts()
        min_count = class_counts.min()

        # Undersample each class to the size of the minority class
        df_resampled = pd.concat([
            df[df[target_column] == cls].sample(n=min_count, replace=False, random_state=42)
            for cls in class_counts.index
        ], ignore_index=True)

        return df_resampled
    
# Concrete Strategy for Hybrid Resampling (SMOTE + Undersampling)
class HybridResamplingStrategy(ResamplingStrategy):
    def __init__(self, smote_ratio=0.5):
        """
        Initializes the HybridResamplingStrategy.

        Parameters:
        smote_ratio (float): The desired ratio of the minority to majority class after SMOTE oversampling.
        """
        self.smote_ratio = smote_ratio

    def resample(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Applies SMOTE to oversample the minority class and then undersamples the majority class.

        Parameters:
        df (pd.DataFrame): The input DataFrame with the target variable.
        target_column (str): The name of the target column.

        Returns:
        pd.DataFrame: The hybrid resampled DataFrame.
        """
        X = df.drop(columns=[target_column])  # Feature columns
        y = df[target_column].astype(int)  # Target column

        # Define SMOTE + Undersampling pipeline
        over = SMOTE(sampling_strategy=self.smote_ratio, random_state=42)
        under = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        pipeline = Pipeline([('smote', over), ('under', under)])

        # Apply resampling
        X_resampled, y_resampled = pipeline.fit_resample(X, y)

        # Convert back to DataFrame
        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled[target_column] = y_resampled

        return df_resampled
    
class Resampler:
    def __init__(self, strategy: ResamplingStrategy):
        """
        Initializes the Resampler with a specific resampling strategy.

        Parameters:
        strategy (ResamplingStrategy): The strategy to be used for handling class imbalance.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ResamplingStrategy):
        """
        Sets a new strategy for the Resampler.

        Parameters:
        strategy (ResamplingStrategy): The new strategy to be used for handling class imbalance.
        """
        self._strategy = strategy

    def resample(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Executes resampling using the current strategy.

        Parameters:
        df (pd.DataFrame): The input DataFrame with the target variable.
        target_column (str): The name of the target column.

        Returns:
        pd.DataFrame: The undersampled DataFrame.
        """
        return self._strategy.resample(df, target_column)