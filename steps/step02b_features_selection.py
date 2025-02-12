import os
import pandas as pd

script_directory = os.path.dirname(__file__)
home_directory = os.path.abspath(os.path.join(script_directory, '..'))

import sys
sys.path.append(home_directory)

from scripts.preprocessing import PreProcessor, DataTypesConversionStrategy
from utils.utils import timetracker

class FeaturesSelection:
    def __init__(self):
        self.input_df_path = os.path.join(home_directory, os.path.join('data', 'master_features_data.csv'))
        self.output_df_path = os.path.join(home_directory, os.path.join('data', 'selected_features_data.csv'))
    
    @timetracker
    def run(self):

        # Load the master features data
        df = pd.read_csv(self.input_df_path)
        
        # Type conversion
        preprocessor = PreProcessor(DataTypesConversionStrategy(object, ['fradulent_user_flag', 'weekend_flag', 'past_fraud_tx_company', 'past_fraud_tx_user', 'past_fraud_tx_card', 'past_fraud_tx_vendor']))
        df = preprocessor.execute(df)
        
        # Defining features to be removed - based on correlations (numerical) and EDA vs target variable (categorical)
        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numerical_features_to_be_removed = \
            [x for x in numerical_features if x.endswith('_card') or x.endswith('_user')] + \
            ['past_tx_count_vendor', 'past_tx_fail_rate_vendor', 'time_since_vendor_added']
            
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        categorical_features_to_be_removed = ['past_fraud_tx_card', 'past_fraud_tx_user', 'past_fraud_tx_vendor']
        
        # Removal of features
        df = df[[x for x in df.columns if x not in numerical_features_to_be_removed + categorical_features_to_be_removed]]

        # Selection of features based on statistical methods - See EDA Jupyter notebook

        # Categorical features encoding
        df['status_settled'] = df['status'].map({'failed': 0, 'settled': 1})
        df['vendor_status_manual_pending'] = df['vendor_status'].isin(['manual_approval', 'pending']).astype(int)
        df['user_gst_available'] = df['user_kyc_category'].isin(['GST only', 'Both']).astype(int)
        
        # Selecting features
        numerical_features = ['past_tx_count_company', 'past_tx_fail_rate_company', 'time_since_last_tx_company', 
                              'unusual_activity_score_vendor', 'past_unique_card_count_company', 'paid_at_time', 
                              'user_aadhar_pan_similarity', 'amount']
        categorical_features = ['weekend_flag', 'past_fraud_tx_company', 'status_settled', 'vendor_status_manual_pending', 
                                'user_gst_available']
        
        df = df[['transaction_uuid'] + numerical_features + categorical_features + ['fradulent_user_flag']]

        # Save selected features dataset
        df.to_csv(self.output_df_path, index=False)

        print('Feature selection step completed')
        
if __name__ == '__main__':
    FeaturesSelection().run()