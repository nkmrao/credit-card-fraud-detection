import pandas as pd
import numpy as np
import joblib
import os
from fuzzywuzzy import fuzz

from scripts.preprocessing import PreProcessor, DataTypesConversionStrategy, SubstringReplacementStrategy, FullStringReplacementStrategy

class Predict:
    def __init__(self):
        
        self.raw_data_path = os.path.join('data', 'raw_data.csv')
        self.pipeline_file_path = os.path.join('models', 'rf_pipeline.pkl')
        
        self.int_cols = []
        self.float_cols = ['amount']
        self.str_cols = ['user_aadhar_name', 'vendor_display_name', 'vendor_bank_account_name', 'vendor_gst_name', 'vendor_pan_name', 'vendor_gst_trade_name', 'user_pan_name', 'user_gst_name']
        self.obj_cols = ['company_uuid', 'transaction_uuid', 'mode', 'vendor_uuid', 'status', 'card_uuid', 'user_uuid', 'kyc_status', 'vendor_status', 'fradulent_user_flag']
        self.timestamp_cols = ['paid_at', 'card_time_of_addition', 'vendor_addition_date']
        
        self.new_row = {'company_uuid': '01065873-3d25-430e-8c23-368a777b58b0',
                         'transaction_uuid': 'a6f44453-6f30-44fb-8c53-66b0aa7d3ede',
                         'mode': 'credit_card',
                         'amount': '2,300',
                         'entry_time': 'May 15, 2024, 3:25 AM',
                         'paid_at': 'May 15, 2024, 3:25 AM',
                         'vendor_uuid': '90678e1b-d6ef-43de-aef6-18e7ed748151',
                         'status': 'settled',
                         'card_uuid': '10ec1592-0e8d-4b0a-bbc9-761112435206',
                         'card_time_of_addition': 'February 23, 2024, 10:37 AM',
                         'user_uuid': 'fa2f6b6a-2b7b-410b-87ec-3cca867c9c55',
                         'kyc_status': 'success',
                         'user_aadhar_name': 'ABDUL KHADER NASHEEL',
                         'vendor_display_name': 'Premier Plastic Industries',
                         'vendor_bank_account_name': 'PRIMIEREPLASTICINDUSTRIES',
                         'vendor_gst_name': 'NAZEER ABDUL HUSSAIN',
                         'vendor_pan_name': np.nan,
                         'vendor_gst_trade_name': 'PREMIER PLASTIC INDUSTRIES',
                         'vendor_status': 'approved',
                         'vendor_addition_date': 'February 21, 2024, 4:56 AM',
                         'user_pan_name': 'VIVEK JAYAWANT DHERE',
                         'user_gst_name': np.nan
                         }
        
        self.proceed = True
    
    def preprocessing(self):
        
        raw_data = pd.read_csv(self.raw_data_path)
        
        # Remove commas from amount
        preprocessor = PreProcessor(SubstringReplacementStrategy('amount', ',',''))
        raw_data = preprocessor.execute(raw_data)
        self.new_row['amount'] = self.new_row['amount'].replace(',', '')
        
        # Replace NaN and blank string values with None
        for col in raw_data.columns:
            if pd.api.types.is_object_dtype(raw_data[col]):
                
                preprocessor.set_strategy(FullStringReplacementStrategy(col, np.nan, None))
                raw_data = preprocessor.execute(raw_data)
                
                preprocessor.set_strategy(FullStringReplacementStrategy(col, '', None))
                raw_data = preprocessor.execute(raw_data)
                
        self.new_row = {k:v if v not in ['', np.nan] else None for k,v in self.new_row.items()}
        
        # Type casting
        
        preprocessor.set_strategy(DataTypesConversionStrategy(int, self.int_cols))
        raw_data = preprocessor.execute(raw_data)
        self.new_row = {k:int(v) if k in self.int_cols else v for k,v in self.new_row.items()}
        
        preprocessor.set_strategy(DataTypesConversionStrategy(float, self.float_cols))
        raw_data = preprocessor.execute(raw_data)
        self.new_row = {k:float(v) if k in self.float_cols else v for k,v in self.new_row.items()}
        
        preprocessor.set_strategy(DataTypesConversionStrategy(str, self.str_cols))
        raw_data = preprocessor.execute(raw_data)
        self.new_row = {k:str(v) if k in self.str_cols else v for k,v in self.new_row.items()}

        preprocessor.set_strategy(DataTypesConversionStrategy(object, self.obj_cols))
        raw_data = preprocessor.execute(raw_data)

        preprocessor.set_strategy(DataTypesConversionStrategy(pd.Timestamp, self.timestamp_cols))
        raw_data = preprocessor.execute(raw_data)        
        self.new_row = {k:pd.Timestamp(v) if k in self.timestamp_cols else v for k,v in self.new_row.items()}
        
        # Remove rows where kyc_status is not 'success'
        raw_data = raw_data.loc[raw_data['kyc_status'] == 'success',:].reset_index(drop=True)
        
        # Sort the dataframe by the time of transaction
        preprocessed_data = raw_data.sort_values('paid_at').reset_index(drop=True)
        
        if self.new_row['kyc_status'] not in 'success':
            print('Invalid KYC status. Unable to make prediction.')
            self.proceed = False
            
        return preprocessed_data
    
    def create_features(self, preprocessed_data):
        
        def compute_percentile(numbers, target):
            
            if not numbers:
                return 0
            
            count = sum(1 for num in numbers if num <= target)
            percentile = (count / len(numbers)) * 100
                        
            return percentile
        
        def similarity_between_strings(str1, str2):
            similarity = fuzz.ratio(str1, str2) / 100
            return similarity
        
        if self.proceed:
            
            self.numerical_features = ['past_tx_count_company', 'past_tx_fail_rate_company', 'time_since_last_tx_company', 
                                       'unusual_activity_score_vendor', 'past_unique_card_count_company', 'paid_at_time', 
                                       'user_aadhar_pan_similarity', 'amount']
            self.categorical_features = ['weekend_flag', 'past_fraud_tx_company', 
                                         'status_settled', 'vendor_status_manual_pending', 'user_gst_available']
            
            past_company_df = preprocessed_data.loc[np.logical_and(preprocessed_data['company_uuid'] == self.new_row['company_uuid'], 
                                                                   preprocessed_data['paid_at'] < self.new_row['paid_at']),:].reset_index(drop=True)
            
            past_tx_count_company = len(past_company_df)
            past_tx_fail_rate_company = (past_company_df['status'] == 'failed').sum() / past_tx_count_company if past_tx_count_company > 0 else 0
            last_tx_time = past_company_df.iloc[-1]['paid_at'] if past_tx_count_company > 0 else None
            time_since_last_tx_company = (self.new_row['paid_at'] - last_tx_time).days + (self.new_row['paid_at'] - last_tx_time).seconds/(24*60*60) if last_tx_time is not None else 0
            past_unique_card_count_company = past_company_df['card_uuid'].nunique()
            past_fraud_tx_company = int(past_company_df['fradulent_user_flag'].sum() > 0)
            
            preprocessed_data['paid_at_date'] = preprocessed_data['paid_at'].apply(lambda x: x.date())
            
            past_vendor_df = preprocessed_data.loc[np.logical_and(preprocessed_data['vendor_uuid'] == self.new_row['vendor_uuid'], 
                                                                   preprocessed_data['paid_at'] < self.new_row['paid_at']),:].reset_index(drop=True)
            past_1d_tx_count_vendor = len(past_vendor_df.loc[past_vendor_df['paid_at'].apply(lambda x: x.date()) == self.new_row['paid_at'].date(),:])
            tx_count_by_date = past_vendor_df.groupby(['vendor_uuid', 'paid_at_date'])['transaction_uuid'].count().reset_index().rename(columns={'transaction_uuid': 'tx_count'})
            prev_day_tx_counts_vendor = tx_count_by_date.loc[np.logical_and(tx_count_by_date['vendor_uuid'] == self.new_row['vendor_uuid'], 
                                                                            tx_count_by_date['paid_at_date'] < self.new_row['paid_at'].date()),:]['tx_count'].tolist()
            tx_freq_percentile_vendor = compute_percentile(prev_day_tx_counts_vendor, past_1d_tx_count_vendor)
            unusual_activity_score_vendor = tx_freq_percentile_vendor * past_1d_tx_count_vendor
            
            user_aadhar_pan_similarity = similarity_between_strings(self.new_row['user_aadhar_name'].lower(), self.new_row['user_pan_name'].lower()) if all(x is not None for x in [self.new_row['user_aadhar_name'], self.new_row['user_pan_name']]) else 0
            
            weekend_flag = int(self.new_row['paid_at'].dayofweek in [5,6])
            status_settled = int(self.new_row['status'] == 'settled')
            vendor_status_manual_pending = int(self.new_row['vendor_status'] != 'approved')
            user_gst_available = int(self.new_row['user_gst_name'] is not None)
            
            paid_at_time = self.new_row['paid_at'].hour*3600 + self.new_row['paid_at'].minute*60 + self.new_row['paid_at'].second
            
            new_data_features = {
                'past_tx_count_company': past_tx_count_company, 
                'past_tx_fail_rate_company': past_tx_fail_rate_company, 
                'time_since_last_tx_company': time_since_last_tx_company, 
                'unusual_activity_score_vendor': unusual_activity_score_vendor, 
                'past_unique_card_count_company': past_unique_card_count_company, 
                'paid_at_time': paid_at_time, 
                'user_aadhar_pan_similarity': user_aadhar_pan_similarity, 
                'amount': self.new_row['amount'], 
                'weekend_flag': weekend_flag, 
                'past_fraud_tx_company': past_fraud_tx_company, 
                'status_settled': status_settled, 
                'vendor_status_manual_pending': vendor_status_manual_pending, 
                'user_gst_available': user_gst_available
            }
            
            return pd.DataFrame([new_data_features])
        
        else:
            return pd.DataFrame()
            
    def predict(self):
        
        preprocessed_data = self.preprocessing()
        if self.proceed:
            new_features_df = self.create_features(preprocessed_data)
            
            loaded_pipeline = joblib.load(self.pipeline_file_path)
            prediction_prob = loaded_pipeline.predict_proba(new_features_df)[0][1]
            print(f'The new transaction has a fraud probability of {prediction_prob*100}%')
    
if __name__ == '__main__':
    Predict().predict()