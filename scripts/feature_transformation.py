import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# Abstract Base Class for Feature Transformation Strategy
# ----------------------------------------------------
# This class defines a common interface for different feature transformation strategies.
# Subclasses must implement the apply_transformation method.
class FeatureTransformationStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to apply feature Transformation transformation to the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: A dataframe with the applied transformations.
        """
        pass


# Concrete Strategy for Log Transformation
# ----------------------------------------
# This strategy applies a logarithmic transformation to skewed features to normalize the distribution.
class LogTransformation(FeatureTransformationStrategy):
    def __init__(self, features):
        """
        Initializes the LogTransformation with the specific features to transform.

        Parameters:
        features (list): The list of features to apply the log transformation to.
        """
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a log transformation to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with log-transformed features.
        """
        logging.info(f"Applying log transformation to features: {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(
                df[feature]
            )  # log1p handles log(0) by calculating log(1+x)
        logging.info("Log transformation completed.")
        return df_transformed


# Concrete Strategy for Standard Scaling
# --------------------------------------
# This strategy applies standard scaling (z-score normalization) to features, centering them around zero with unit variance.
class StandardScaling(FeatureTransformationStrategy):
    def __init__(self, features):
        """
        Initializes the StandardScaling with the specific features to scale.

        Parameters:
        features (list): The list of features to apply the standard scaling to.
        """
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies standard scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with scaled features.
        """
        logging.info(f"Applying standard scaling to features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Standard scaling completed.")
        return df_transformed


# Concrete Strategy for Min-Max Scaling
# -------------------------------------
# This strategy applies Min-Max scaling to features, scaling them to a specified range, typically [0, 1].
class MinMaxScaling(FeatureTransformationStrategy):
    def __init__(self, features, feature_range=(0, 1)):
        """
        Initializes the MinMaxScaling with the specific features to scale and the target range.

        Parameters:
        features (list): The list of features to apply the Min-Max scaling to.
        feature_range (tuple): The target range for scaling, default is (0, 1).
        """
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Min-Max scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with Min-Max scaled features.
        """
        logging.info(
            f"Applying Min-Max scaling to features: {self.features} with range {self.scaler.feature_range}"
        )
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Min-Max scaling completed.")
        return df_transformed


# Concrete Strategy for One-Hot Encoding
# --------------------------------------
# This strategy applies one-hot encoding to categorical features, converting them into binary vectors.
class OneHotEncoding(FeatureTransformationStrategy):
    def __init__(self, features):
        """
        Initializes the OneHotEncoding with the specific features to encode.

        Parameters:
        features (list): The list of categorical features to apply the one-hot encoding to.
        """
        self.features = features
        self.encoder = OneHotEncoder(sparse=False, drop="first")

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies one-hot encoding to the specified categorical features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with one-hot encoded features.
        """
        logging.info(f"Applying one-hot encoding to features: {self.features}")
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )
        df_transformed = df_transformed.drop(columns=self.features).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        logging.info("One-hot encoding completed.")
        return df_transformed


# Context Class for Feature Transformation
# -------------------------------------
# This class uses a FeatureTransformationStrategy to apply transformations to a dataset.
class FeatureTransform:
    def __init__(self, strategy: FeatureTransformationStrategy):
        """
        Initializes the FeatureTransform with a specific feature Transformation strategy.

        Parameters:
        strategy (FeatureTransformationStrategy): The strategy to be used for feature Transformation.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureTransformationStrategy):
        """
        Sets a new strategy for the FeatureTransform.

        Parameters:
        strategy (FeatureTransformationStrategy): The new strategy to be used for feature Transformation.
        """
        logging.info("Switching feature Transformation strategy.")
        self._strategy = strategy

    def apply_feature_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the feature Transformation transformation using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with applied feature Transformation transformations.
        """
        logging.info("Applying feature Transformation strategy.")
        return self._strategy.apply_transformation(df)