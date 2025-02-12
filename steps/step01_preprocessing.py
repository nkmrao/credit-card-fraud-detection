import os
import pandas as pd
import numpy as np

script_directory = os.path.dirname(__file__)
home_directory = os.path.abspath(os.path.join(script_directory, '..'))

import sys
sys.path.append(home_directory)

from scripts.preprocessing import PreProcessor, DataTypesConversionStrategy, SubstringReplacementStrategy, FullStringReplacementStrategy
from utils.utils import timetracker

class PreProcessorStep:
    def __init__(self):
        self.input_df_path = os.path.join(home_directory, os.path.join('data', 'raw_data.csv'))
        self.output_df_path = os.path.join(home_directory, os.path.join('data', 'preprocessed_data.csv'))
    
    @timetracker
    def run(self):
        
        # Load the raw data
        df = pd.read_csv(self.input_df_path)

        # Remove commas from amount column
        preprocessor = PreProcessor(SubstringReplacementStrategy('amount', ',',''))
        df = preprocessor.execute(df)

        # Replace NaN and blank string values with None
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                
                preprocessor.set_strategy(FullStringReplacementStrategy(col, np.nan, None))
                df = preprocessor.execute(df)
                
                preprocessor.set_strategy(FullStringReplacementStrategy(col, '', None))
                df = preprocessor.execute(df)

        # Type cast each column to the appropriate data type
        int_cols = []
        float_cols = ['amount']
        str_cols = ['user_aadhar_name', 'vendor_display_name', 'vendor_bank_account_name', 'vendor_gst_name', 'vendor_pan_name', 'vendor_gst_trade_name', 'user_pan_name', 'user_gst_name']
        obj_cols = ['company_uuid', 'transaction_uuid', 'mode', 'vendor_uuid', 'status', 'card_uuid', 'user_uuid', 'kyc_status', 'vendor_status', 'fradulent_user_flag']
        timestamp_cols = ['entry_time', 'paid_at', 'card_time_of_addition', 'vendor_addition_date']

        preprocessor.set_strategy(DataTypesConversionStrategy(int, int_cols))
        df = preprocessor.execute(df)

        preprocessor.set_strategy(DataTypesConversionStrategy(float, float_cols))
        df = preprocessor.execute(df)

        preprocessor.set_strategy(DataTypesConversionStrategy(str, str_cols))
        df = preprocessor.execute(df)

        preprocessor.set_strategy(DataTypesConversionStrategy(object, obj_cols))
        df = preprocessor.execute(df)

        preprocessor.set_strategy(DataTypesConversionStrategy(pd.Timestamp, timestamp_cols))
        df = preprocessor.execute(df)
        
        # Remove rows where kyc_status is not 'success'
        df = df.loc[df['kyc_status'] == 'success',:].reset_index(drop=True)
        
        # Remove unnecessary variables 
        df = df[[x for x in df.columns if x not in ['mode', 'kyc_status', 'entry_time', 'vendor_display_name', 'vendor_bank_account_name']]]
        
        # Save preprocessed df
        df.to_csv(self.output_df_path, index=False)
        
        print('Preprocessing step completed')

if __name__ == '__main__':
    PreProcessorStep().run()