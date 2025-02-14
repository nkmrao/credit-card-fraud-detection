from abc import ABC, abstractmethod

import pandas as pd
from typing import List, Type

# Abstract Base Class for Preprocessing Strategies
class PreProcessingStrategy(ABC):
    @abstractmethod
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform specific type of data manipulation.

        Parameters:
        df (pd.DataFrame): The dataframe on which the data manipulation is to be performed.

        Returns:
        df (pd.DataFrame): The modified dataframe after executing the preprocessing strategy.
        """
        pass

# Concrete Strategy for Type Casting
# --------------------------------------------
# This strategy converts the data types of given columns to numerical or categorical as specified.
class DataTypesConversionStrategy(PreProcessingStrategy):
    def __init__(self, target_type: Type, cols_li: List[str]):
        """
        Initializes the class with given list of columns and target_type.

        Parameters:
        target_type: The new target_type to be used for type casting.
        cols_li: The new list of columns whose type is to be converted.

        Returns:
        None
        """

        if target_type not in {float, int, str, object, pd.Timestamp}:
            raise ValueError(f"Invalid target_type '{target_type}'. Choose from: float, int, str, object or pd.Timestamp.")

        self._target_type = target_type
        self._cols_li = cols_li

    def set_inputs(self, target_type: Type, cols_li: List[str]):
        """
        Sets new inputs for the class.

        Parameters:
        target_type: The new target_type to be used for type casting.
        cols_li: The new list of columns whose type is to be converted.

        Returns:
        None
        """

        if target_type not in {float, int, str, object}:
            raise ValueError(f"Invalid target_type '{target_type}'. Choose from: float, int, str, object.")

        self._target_type = target_type
        self._cols_li = cols_li

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts the data type of the given list of columns to the target_type.

        Parameters:
        df (pd.DataFrame): The dataframe to be manipulated.

        Returns:
        df (pd.DataFrame): The modified dataframe after type casting.
        """
        if len(self._cols_li) > 0:
            if self._target_type == pd.Timestamp:
                df[self._cols_li] = df[self._cols_li].apply(pd.to_datetime, errors='coerce').where(df[self._cols_li].notna(), None)
            else:
                df[self._cols_li] = df[self._cols_li].astype(self._target_type, errors='ignore').where(df[self._cols_li].notna(), None)
        
        return df
    
class SubstringReplacementStrategy(PreProcessingStrategy):
    def __init__(self, col: str, substr: str, target_substr: str):
        """
        Initializes the class with column name, and from and to substrings.

        Parameters:
        col: The column in which replacement is to be done
        substr: The substring to be replaced (from substring).
        target_substr: The target substring after replacement (to substring).

        Returns:
        None
        """
        self._col = col
        self._substr = substr
        self._target_substr = target_substr

    def set_inputs(self, col: str, substr: str, target_substr: str):
        """
        Sets new inputs for the class.

        Parameters:
        col: The column in which replacement is to be done
        substr: The substring to be replaced (from substring).
        target_substr: The target substring after replacement (to substring).

        Returns:
        None
        """
        self._col = col
        self._substr = substr
        self._target_substr = target_substr

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces the given substring with the given target substring in the given column.

        Parameters:
        df (pd.DataFrame): The dataframe to be manipulated.

        Returns:
        df (pd.DataFrame): The modified dataframe after type casting.
        """
        if self._col not in df.columns:
            raise ValueError(f"Column '{self._col}' not found in DataFrame")

        # Check if the column is of type str or object
        if not pd.api.types.is_object_dtype(df[self._col]):
            raise TypeError(f"Column '{self._col}' must be of type 'str' or 'object'")

        # Replace the substring in the specified column
        df[self._col] = df[self._col].str.replace(self._substr, self._target_substr, regex=False)

        return df

class FullStringReplacementStrategy(PreProcessingStrategy):
    def __init__(self, col: str, string: str, target: str|None):
        """
        Initializes the class with column name, and from and to values.

        Parameters:
        col: The column in which replacement is to be done
        string: The string to be replaced (from string).
        target: The target value after replacement (to string).

        Returns:
        None
        """
        self._col = col
        self._string = string
        self._target = target

    def set_inputs(self, col: str, string: str, target: str|None):
        """
        Sets new inputs for the class.

        Parameters:
        col: The column in which replacement is to be done
        string: The string to be replaced (from string).
        target: The target value after replacement (to string).

        Returns:
        None
        """
        self._col = col
        self._string = string
        self._target = target

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces the given string with the given target in the given column.

        Parameters:
        df (pd.DataFrame): The dataframe to be manipulated.

        Returns:
        df (pd.DataFrame): The modified dataframe after type casting.
        """
        if self._col not in df.columns:
            raise ValueError(f"Column '{self._col}' not found in DataFrame")

        # Check if the column is of type str or object
        if not pd.api.types.is_object_dtype(df[self._col]):
            raise TypeError(f"Column '{self._col}' must be of type 'str' or 'object'")

        # Replace the substring in the specified column
        df[self._col] = df[self._col].replace(self._string, self._target)

        return df

class PreProcessor:
    def __init__(self, strategy: PreProcessingStrategy):
        """
        Initializes the PreProcessor with a specific preprocessing strategy.

        Parameters:
        strategy (PreProcessingStrategy): The strategy to be used for data manipulation.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: PreProcessingStrategy):
        """
        Sets a new strategy for the class.

        Parameters:
        strategy (PreProcessingStrategy): The new strategy to be used for data manipulation.

        Returns:
        None
        """
        self._strategy = strategy

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes preprocessing using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be manipulated.

        Returns:
        df (pd.DataFrame): The manipulated dataframe after executing the preprocessing strategy.
        """
        df = self._strategy.execute(df)

        return df