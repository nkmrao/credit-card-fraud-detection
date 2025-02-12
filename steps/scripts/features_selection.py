from abc import ABC, abstractmethod

import pandas as pd
from typing import List
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2
from sklearn.preprocessing import LabelEncoder

# Abstract Base Class for Feature Selection Strategies
class FeatureSelectionStrategy(ABC):
    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Abstract method to evaluate the feature selection method.

        Parameters:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Binary target variable.

        Returns:
        pd.Series: Evaluation scores for each feature.
        """
        pass

    @abstractmethod
    def select(self, X: pd.DataFrame, y: pd.Series, threshold: float) -> List[str]:
        """
        Selects features based on  score.

        Parameters:
        X (pd.DataFrame): Feature set with categorical features.
        y (pd.Series): Binary target variable.
        threshold (float): Minimum Mutual Information score required to keep a feature.

        Returns:
        List[str]: list of selected features names.
        """
        pass

# Concrete Strategy for Mutual Information
# --------------------------------------------
# This strategy implements the Mutual Information statistical method to select features.
class MutualInformationStrategy(FeatureSelectionStrategy):
    def __init__(self, X_type):
        """
        Initializes the class with type of X features.

        Parameters:
        X_type: Whether the strategy is for 'numerical' or 'categorical' features.

        Returns:
        None
        """
        self._X_type = X_type

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Evaluates features based on Mutual Information (MI) score.

        Parameters:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Binary target variable.

        Returns:
        pd.Series: MI scores for each feature. Higher the score, the feature is more likely to be important.
        """
        if self._X_type == 'numerical':
            mi_scores = mutual_info_classif(X, y, discrete_features=False)
            return pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        else:
            X_encoded = pd.get_dummies(X, drop_first=True)
            mi_scores = mutual_info_classif(X_encoded, y, discrete_features=True)
            return pd.Series(mi_scores, index=X_encoded.columns).sort_values(ascending=False)
        
    def select(self, X: pd.DataFrame, y: pd.Series, threshold: float) -> List[str]:
        """
        Selects features based on Mutual Information (MI) score and the given threshold.

        Parameters:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Binary target variable.
        threshold (float): Minimum Mutual Information score required to keep a feature.

        Returns:
        List[str]: List of feature names that are selected as per the threshold
        """
        if self._X_type == 'numerical':
            mi_scores = mutual_info_classif(X, y, discrete_features=False)
            mi_scores_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
            return list(mi_scores_series.index)
        else:
            X_encoded = pd.get_dummies(X, drop_first=True)
            mi_scores = mutual_info_classif(X_encoded, y, discrete_features=True)
            mi_scores_series = pd.Series(mi_scores, index=X_encoded.columns).sort_values(ascending=False)
            return list(mi_scores_series.index)
        

# Concrete Strategy for ANOVA F-value
# --------------------------------------------
# This strategy implements the ANOVA (F-score) statistical method to select numerical features.
class ANOVAFscoreStrategy(FeatureSelectionStrategy):
    def __init__(self):
        """
        Initializes the class

        Parameters:
        None

        Returns:
        None
        """
        pass

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Evaluates features based on F-score.

        Parameters:
        X (pd.DataFrame): Feature set of numerical variables.
        y (pd.Series): Binary target variable.

        Returns:
        pd.Series: F-scores for each feature. Higher the score, the feature is more likely to be important.
        """
        f_scores, p_values = f_classif(X, y)
        return pd.Series(f_scores, index=X.columns).sort_values(ascending=False)
        
    def select(self, X: pd.DataFrame, y: pd.Series, threshold: float) -> List[str]:
        """
        Selects features based on F-score and the given threshold.

        Parameters:
        X (pd.DataFrame): Feature set of numerical variables.
        y (pd.Series): Binary target variable.
        threshold (float): Minimum F-score required to keep a feature.

        Returns:
        List[str]: List of feature names that are selected as per the threshold
        """
        f_scores, p_values = f_classif(X, y)
        f_scores_series = pd.Series(f_scores, index=X.columns).sort_values(ascending=False)
        return list(f_scores_series[f_scores_series > threshold].index)
    
# Concrete Strategy for Chi-Square Test
# --------------------------------------------
# This strategy implements the Chi-Square Test statistical method to select categorical features.
class ChiSquareStrategy(FeatureSelectionStrategy):
    def __init__(self):
        """
        Initializes the class

        Parameters:
        None

        Returns:
        None
        """
        pass

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Evaluates features based on Chi-Square Test.

        Parameters:
        X (pd.DataFrame): Feature set of categorical variables.
        y (pd.Series): Binary target variable.

        Returns:
        pd.Series: Chi-scores for each feature. Higher the score, the feature is more likely to be important.
        """
        X_encoded = X.apply(LabelEncoder().fit_transform)
        chi_scores, p_values = chi2(X_encoded, y)
        return pd.Series(chi_scores, index=X.columns).sort_values(ascending=False)
        
    def select(self, X: pd.DataFrame, y: pd.Series, threshold: float) -> List[str]:
        """
        Selects features based on Chi-Square Test and the given threshold.

        Parameters:
        X (pd.DataFrame): Feature set of categorical variables.
        y (pd.Series): Binary target variable.
        threshold (float): Minimum F-score required to keep a feature.

        Returns:
        List[str]: List of feature names that are selected as per the threshold
        """
        X_encoded = X.apply(LabelEncoder().fit_transform)
        chi_scores, p_values = chi2(X_encoded, y)
        chi_scores_series = pd.Series(chi_scores, index=X.columns).sort_values(ascending=False)
        return list(chi_scores_series[chi_scores_series > threshold].index)
    
class FeaturesSelector:
    def __init__(self, strategy: FeatureSelectionStrategy):
        """
        Initializes the FeaturesSelector with a specific strategy.

        Parameters:
        strategy (FeatureSelectionStrategy): The strategy to be used for selecting features.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureSelectionStrategy):
        """
        Sets a new strategy for the class.

        Parameters:
        strategy (FeatureSelectionStrategy): The new strategy to be used for selecting features.

        Returns:
        None
        """
        self._strategy = strategy

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Evaluates features based on the selected technique.

        Parameters:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Binary target variable.

        Returns:
        pd.Series: Scores for each feature.
        """
        return self._strategy.evaluate(X, y)
    
    def select(self, X: pd.DataFrame, y: pd.Series, threshold: float) -> List[str]:
        """
        Selects features based the selected technique and the given threshold.

        Parameters:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Binary target variable.
        threshold (float): Minimum F-score required to keep a feature.

        Returns:
        List[str]: List of feature names that are selected as per the threshold
        """
        return self._strategy.select(X, y, threshold)