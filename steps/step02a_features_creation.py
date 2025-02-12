import os
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz

script_directory = os.path.dirname(__file__)
home_directory = os.path.abspath(os.path.join(script_directory, '..'))

import sys
sys.path.append(home_directory)

from scripts.preprocessing import PreProcessor, DataTypesConversionStrategy, SubstringReplacementStrategy, FullStringReplacementStrategy
from utils.utils import timetracker

class FeaturesCreation:
    def __init__(self):
        self.input_df_path = os.path.join(home_directory, os.path.join('data', 'preprocessed_data.csv'))
        self.output_df_path = os.path.join(home_directory, os.path.join('data', 'master_features_data.csv'))
            
    def compute_tx_related_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df['paid_at_date'] = df['paid_at'].dt.date
        df['paid_at_time'] = df['paid_at'].dt.time.apply(lambda x: x.hour*3600 + x.minute*60 + x.second)
        df['weekend_flag'] = df['paid_at'].dt.dayofweek.isin([5, 6]).astype(int).astype(object)
        
        return df
    
    def compute_past_tx_behaviour_features(self, df: pd.DataFrame, groupby_col='company_uuid') -> pd.DataFrame:
        
        suffix = groupby_col.replace('_uuid', '')
        
        df[f'past_tx_count_{suffix}'] = df.groupby(groupby_col).cumcount()
        df[f'past_fraud_tx_{suffix}'] = df.groupby(groupby_col)['fradulent_user_flag'].cummax().shift(1, fill_value=0)
        
        df[f'past_tx_fail_count_{suffix}'] = df.groupby(groupby_col)['status'].transform(lambda x: (x == 'failed').cumsum().shift(1, fill_value=0))
        df[f'past_tx_fail_rate_{suffix}'] = df[f'past_tx_fail_count_{suffix}'] / df[f'past_tx_count_{suffix}']
        df[f'past_tx_fail_rate_{suffix}'] = df[f'past_tx_fail_rate_{suffix}'].fillna(0)
        
        df[f'time_since_last_tx_{suffix}'] = df.groupby(groupby_col)['paid_at'].diff()
        df[f'time_since_last_tx_{suffix}'] = df[f'time_since_last_tx_{suffix}'].fillna(pd.Timedelta(seconds=0))
        df[f'time_since_last_tx_{suffix}'] = df[f'time_since_last_tx_{suffix}'].apply(lambda x: x.days + x.seconds/(24*60*60))
        
        def past_1_day_tx_count(group):
            return group.rolling('1D').count() - 1
        
        def compute_percentile(numbers, target):
            
            if not numbers:
                return 0
            
            count = sum(1 for num in numbers if num <= target)
            percentile = (count / len(numbers)) * 100
                        
            return percentile
        
        df.set_index('paid_at', inplace=True)
        df[f'past_1_day_tx_count_{suffix}'] = df.groupby(groupby_col)['transaction_uuid'].transform(past_1_day_tx_count)
        df.reset_index(inplace=True)
        
        tx_count_by_date = df.groupby([groupby_col, 'paid_at_date'])['transaction_uuid'].count().reset_index().rename(columns={'transaction_uuid': 'tx_count'})
        df[f'prev_day_tx_counts_{suffix}'] = df.apply(lambda row: tx_count_by_date.loc[np.logical_and(tx_count_by_date[groupby_col] == row[groupby_col], 
                                                                                            tx_count_by_date['paid_at_date'] < row['paid_at_date']), 'tx_count'].tolist(), axis=1)
        df[f'tx_freq_percentile_{suffix}'] = df.apply(lambda row: compute_percentile(row[f'prev_day_tx_counts_{suffix}'], row[f'past_1_day_tx_count_{suffix}']), axis=1)
        df[f'unusual_activity_score_{suffix}'] = df[f'tx_freq_percentile_{suffix}'] * df[f'past_1_day_tx_count_{suffix}']
        
        def cumulative_unique_cards(group):
            unique_cards = set()
            cumulative_counts = []
            for card in group['card_uuid']:
                unique_cards.add(card)
                cumulative_counts.append(len(unique_cards))
            return cumulative_counts
        
        if suffix in ['company', 'user']:
            col_name = f'past_unique_card_count_{suffix}'
            df = df.groupby(groupby_col, group_keys=False).apply(lambda x: x.assign(**{col_name: cumulative_unique_cards(x)}))
        
        return df
    
    def compute_card_related_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df['card_active_duration'] = (df['paid_at'] - df['card_time_of_addition']).apply(lambda x: x.days + x.seconds/(24*60*60))
        
        return df
    
    def get_kyc_category(self, pan_name, gst_name):
        if pan_name is None:
            if gst_name is None:
                return 'Neither'
            else:
                return 'GST only'
        else:
            if gst_name is None:
                return 'Pan only'
            else:
                return 'Both'
            
    def similarity_between_strings(self, str1, str2):
        similarity = fuzz.ratio(str1, str2) / 100
        return similarity
    
    def compute_user_related_features(self, df: pd.DataFrame) -> pd.DataFrame: 
        df['user_kyc_category'] = df.apply(lambda row: self.get_kyc_category(row['user_pan_name'], row['user_gst_name']), axis=1)
        df['user_aadhar_pan_similarity'] = df.apply(lambda row: self.similarity_between_strings(row['user_aadhar_name'].lower(), row['user_pan_name'].lower()) if row['user_pan_name'] is not None and row['user_aadhar_name'] is not None else 0, axis=1)
        return df
    
    def compute_vendor_related_features(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df['vendor_gst_name_combined'] = df.apply(lambda row: row['vendor_gst_trade_name'] if row['vendor_gst_trade_name'] is not None else row['vendor_gst_name'], axis=1)
        df['vendor_kyc_category'] = df.apply(lambda row: self.get_kyc_category(row['vendor_pan_name'], row['vendor_gst_name_combined']), axis=1)
        
        df['time_since_vendor_added'] = (df['paid_at'] - df['vendor_addition_date']).apply(lambda x: x.days + x.seconds/(24*60*60))
        
        return df
        
    @timetracker
    def run(self):
        
        # Load the preprocessed data
        df = pd.read_csv(self.input_df_path)
        
        preprocessor = PreProcessor(SubstringReplacementStrategy('amount', ',',''))
        
        # Replace NaN and blank string values with None
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                
                preprocessor.set_strategy(FullStringReplacementStrategy(col, np.nan, None))
                df = preprocessor.execute(df)
                
                preprocessor.set_strategy(FullStringReplacementStrategy(col, '', None))
                df = preprocessor.execute(df)

        # Type cast timestamp columns
        timestamp_cols = ['paid_at', 'card_time_of_addition', 'vendor_addition_date']

        preprocessor.set_strategy(DataTypesConversionStrategy(pd.Timestamp, timestamp_cols))
        df = preprocessor.execute(df)
                
        # Sort the dataframe by the time of transaction
        df = df.sort_values('paid_at').reset_index(drop=True)
        
        # Compute tx related features
        df = self.compute_tx_related_features(df)
        df = self.compute_past_tx_behaviour_features(df, groupby_col='company_uuid')
        df = self.compute_past_tx_behaviour_features(df, groupby_col='card_uuid')
        df = self.compute_past_tx_behaviour_features(df, groupby_col='user_uuid')
        df = self.compute_past_tx_behaviour_features(df, groupby_col='vendor_uuid')
        
        # Compute card related features
        df = self.compute_card_related_features(df)
        
        # Compute user related features
        df = self.compute_user_related_features(df)
        
        # Compute vendor related features
        df = self.compute_vendor_related_features(df)
        
        # Features columns
        features_cols = [
            'past_tx_count_company', 'past_fraud_tx_company', 'past_tx_fail_rate_company', 'time_since_last_tx_company', 'unusual_activity_score_company', 
            'past_tx_count_card', 'past_fraud_tx_card', 'past_tx_fail_rate_card', 'time_since_last_tx_card', 'unusual_activity_score_card', 
            'past_tx_count_user', 'past_fraud_tx_user', 'past_tx_fail_rate_user', 'time_since_last_tx_user', 'unusual_activity_score_user', 
            'past_tx_count_vendor', 'past_fraud_tx_vendor', 'past_tx_fail_rate_vendor', 'time_since_last_tx_vendor', 'unusual_activity_score_vendor', 
            'past_unique_card_count_company', 'past_unique_card_count_user', 
            'paid_at_time', 'weekend_flag', 
            'card_active_duration', 
            'user_aadhar_pan_similarity', 'user_kyc_category', 
            'vendor_kyc_category', 'time_since_vendor_added', 'vendor_status',
            'amount', 'status'
        ]
        df = df[['transaction_uuid'] + features_cols + ['fradulent_user_flag']]
        
        # Save master features dataset
        df.to_csv(self.output_df_path, index=False)
        
        print('Features creation step completed')

if __name__ == '__main__':
    df = FeaturesCreation().run()