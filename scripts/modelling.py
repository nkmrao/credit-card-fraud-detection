from abc import ABC, abstractmethod

from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Abstract Base Class for Model Building
class ModellingStrategy(ABC):

    def evaluate_performance(self, y: pd.Series, y_pred: pd.Series) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Common method to evaluate the performance of binary classification models.

        Parameters:
        y (pd.DataFrame): The target variable in the original dataset.
        y_pred (pd.Series): The predictions by the model

        Returns:
        (conf_matrix, perf_scores) - Confusion matrix (nparray) and performance scores (dict)
        """
        conf_matrix = confusion_matrix(y, y_pred)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        perf_scores = {'accuracy': accuracy, 'precision': precision, 'recall': recall}

        return conf_matrix, perf_scores
    
    def plot_confusion_matrix(self, cm, classes, perf_scores, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        
        print(f'Accuracy: {str(round(perf_scores["accuracy"]*100, 2))}%')
        print(f'Precision: {str(round(perf_scores["precision"]*100, 2))}%')
        print(f'Recall: {str(round(perf_scores["recall"]*100, 2))}%')
        print()

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=0)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    @abstractmethod
    def evaluate(self):
        """
        Abstract method to evaluate the performance of a specified model.

        Parameters:
        X (pd.DataFrame): Dataframe of independent variables.
        y (pd.Series): The target variable

        Returns:
        None
        """
        pass

    @abstractmethod
    def feature_importance(self):
        """
        Abstract method to evaluate the importance of features post model training.

        Parameters:
        X (pd.DataFrame): Dataframe of independent variables.
        y (pd.Series): The target variable

        Returns:
        None
        """
        pass

# Concrete Strategy for Logistic Regression
class LogisticRegressionModel(ModellingStrategy):
    def __init__(self, X: pd.DataFrame, y: pd.Series, cv_type: str = 'stratified k-fold'):
        self._X = X
        self._y = y
        
        self.model = LogisticRegression(max_iter=1000)

        self.cv = None
        if cv_type == 'stratified k-fold':
            self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def evaluate(self):
        
        if self.cv is not None:
            y_pred = cross_val_predict(self.model, self._X, self._y, cv=self.cv)
            conf_matrix, perf_scores = self.evaluate_performance(self._y, y_pred)
            self.plot_confusion_matrix(conf_matrix, ['Non-Fraud', 'Fraud'], perf_scores)

    def feature_importance(self):
        
        self.model.fit(self._X, self._y)
        feature_importances = self.model.coef_[0]  # Get coefficients for each feature

        importance_df = pd.DataFrame({
            'Feature': self._X.columns,
            'Coefficient': feature_importances,
            'Absolute_Coefficient': np.abs(feature_importances)  # Taking absolute values for importance ranking
        }).sort_values(by='Absolute_Coefficient', ascending=False)

        # Plot feature importance
        plt.figure(figsize=(6, 10))
        plt.barh(importance_df['Feature'], importance_df['Absolute_Coefficient'], color='skyblue')
        plt.xlabel('Absolute Coefficient Value')
        plt.ylabel('Feature Name')
        plt.title('Feature Importance from Logistic Regression')
        plt.gca().invert_yaxis()  # Invert y-axis to display the most important feature at the top
        plt.show()

# Concrete Strategy for Random Forest
class RandomForestModel(ModellingStrategy):
    def __init__(self, X: pd.DataFrame, y: pd.Series, cv_type: str = 'stratified k-fold'):
        self._X = X
        self._y = y
        
        self.model = RandomForestClassifier(random_state=42)

        self.cv = None
        if cv_type == 'stratified k-fold':
            self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def evaluate(self):
        
        if self.cv is not None:
            y_pred = cross_val_predict(self.model, self._X, self._y, cv=self.cv)
            conf_matrix, perf_scores = self.evaluate_performance(self._y, y_pred)
            self.plot_confusion_matrix(conf_matrix, ['Non-Fraud', 'Fraud'], perf_scores)

    def feature_importance(self):
        
        # Train the model on the entire dataset to get feature importances
        self.model.fit(self._X, self._y)
        feature_importances = self.model.feature_importances_

        feature_importance_df = pd.DataFrame({
            'Feature': self._X.columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        # Plot feature importance
        plt.figure(figsize=(6, 10))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature Name')
        plt.title('Feature Importance from Random Forest Classifier')
        plt.gca().invert_yaxis()  # Invert y-axis to display the most important feature at the top
        plt.show()

# Concrete Strategy for Gradient Boosting
class GradientBoostingModel(ModellingStrategy):
    def __init__(self, X: pd.DataFrame, y: pd.Series, cv_type: str = 'stratified k-fold'):
        self._X = X
        self._y = y
        
        self.model = GradientBoostingClassifier(random_state=42)

        self.cv = None
        if cv_type == 'stratified k-fold':
            self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def evaluate(self):
        
        if self.cv is not None:
            y_pred = cross_val_predict(self.model, self._X, self._y, cv=self.cv)
            conf_matrix, perf_scores = self.evaluate_performance(self._y, y_pred)
            self.plot_confusion_matrix(conf_matrix, ['Non-Fraud', 'Fraud'], perf_scores)

    def feature_importance(self):
        
        # Train the model on the entire dataset to get feature importances
        self.model.fit(self._X, self._y)
        feature_importances = self.model.feature_importances_

        feature_importance_df = pd.DataFrame({
            'Feature': self._X.columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        # Plot feature importance
        plt.figure(figsize=(6, 10))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature Name')
        plt.title('Feature Importance from Gradient Boosting Classifier')
        plt.gca().invert_yaxis()  # Invert y-axis to display the most important feature at the top
        plt.show()