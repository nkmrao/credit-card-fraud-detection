import os
import pandas as pd

script_directory = os.path.dirname(__file__)
home_directory = os.path.abspath(os.path.join(script_directory, '..'))

import sys
sys.path.append(home_directory)

from scripts.preprocessing import PreProcessor, DataTypesConversionStrategy
from scripts.outlier_detection import OutlierDetector, IQROutlierDetection
from scripts.feature_transformation import FeatureTransform, StandardScaling, MinMaxScaling, LogTransformation
from scripts.resampling import Resampler, HybridResamplingStrategy
from utils.utils import timetracker

class FeatureEngineering:
    def __init__(self):
        self.input_df_path = os.path.join(home_directory, os.path.join('data', 'selected_features_data.csv'))
        self.output_df_path = os.path.join(home_directory, os.path.join('data', 'modelling_data.csv'))

    @timetracker
    def run(self):

        # Read selected features data
        df = pd.read_csv(self.input_df_path)

        # Convert types of categorical variables
        categorical_variables = ['fradulent_user_flag', 'weekend_flag', 'past_fraud_tx_company', 
                                 'status_settled', 'vendor_status_manual_pending', 'user_gst_available']
        preprocessor = PreProcessor(DataTypesConversionStrategy(object, categorical_variables))
        df = preprocessor.execute(df)

        numerical_variables = [x for x in df.columns if x not in categorical_variables + ['transaction_uuid']]

        # Cap outliers in numerical variables
        outlier_detector = OutlierDetector(IQROutlierDetection())
        df[numerical_variables] = outlier_detector.handle_outliers(df[numerical_variables], method='cap')
        
        # Transform numerical variables
        standard_scaling_variables = ['paid_at_time']
        log_transform_variables = [x for x in numerical_variables if x not in standard_scaling_variables]

        feature_transformer = FeatureTransform(StandardScaling(standard_scaling_variables))
        df = feature_transformer.apply_feature_transformation(df)

        feature_transformer.set_strategy(LogTransformation(log_transform_variables))
        df = feature_transformer.apply_feature_transformation(df)

        # Minmax scale all numerical variables
        feature_transformer.set_strategy(MinMaxScaling(numerical_variables))
        df = feature_transformer.apply_feature_transformation(df)

        # Resampling of the dataset to balance the classes
        del df['transaction_uuid']
        resampler = Resampler(HybridResamplingStrategy())
        df = resampler.resample(df, 'fradulent_user_flag')
        
        # Save modelling data
        df.to_csv(self.output_df_path, index=False)

        print('Feature engineering step completed')

if __name__ == '__main__':
    FeatureEngineering().run()