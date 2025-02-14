import os
import pandas as pd
import numpy as np
import joblib

script_directory = os.path.dirname(__file__)
home_directory = os.path.abspath(os.path.join(script_directory, '..'))

import sys
sys.path.append(home_directory)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from utils.utils import timetracker

class ModelPipeline:
    def __init__(self):

        self.input_df_path = os.path.join(home_directory, os.path.join('data', 'modelling_data.csv'))

        self.numerical_features = ['past_tx_count_company', 'past_tx_fail_rate_company', 'time_since_last_tx_company', 
                                   'unusual_activity_score_vendor', 'past_unique_card_count_company', 'paid_at_time', 
                                   'user_aadhar_pan_similarity', 'amount']
        self.categorical_features = ['weekend_flag', 'past_fraud_tx_company', 
                                     'status_settled', 'vendor_status_manual_pending', 'user_gst_available']

        self.log_transform_features = ['past_tx_count_company', 'past_tx_fail_rate_company', 'time_since_last_tx_company', 
                                       'unusual_activity_score_vendor', 'past_unique_card_count_company', 
                                       'user_aadhar_pan_similarity', 'amount']
        self.standard_scaler_features = ['paid_at_time']

        self.model_name = 'rf'
        self.model = RandomForestClassifier(random_state=42)

    @timetracker
    def run(self):

        df = pd.read_csv(self.input_df_path)
        
        log_transformer = FunctionTransformer(np.log1p, validate=True)

        preprocessor = ColumnTransformer(
            transformers=[
                ('log', log_transformer, self.log_transform_features),
                ('std_scaler', StandardScaler(), self.standard_scaler_features),
                ('binary', 'passthrough', self.categorical_features)
        ])

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('min_max_scaler', MinMaxScaler()),
            ('classifier', self.model)
        ])

        X = df.drop(columns=['fradulent_user_flag'])
        y = df['fradulent_user_flag']

        pipeline.fit(X, y)

        pipeline_file = os.path.join(home_directory, os.path.join('models', f'{self.model_name}_pipeline.pkl'))
        joblib.dump(pipeline, pipeline_file)
        print(f'Pipeline saved to {pipeline_file}')

if __name__ == '__main__':
    ModelPipeline().run()