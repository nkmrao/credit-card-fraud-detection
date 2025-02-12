import os
import sys

script_directory = os.path.dirname(__file__)
steps_directory = os.path.abspath(os.path.join(script_directory, 'steps'))

sys.path.append(steps_directory)

from steps.step01_preprocessing import PreProcessorStep
from steps.step02a_features_creation import FeaturesCreation
from steps.step02b_features_selection import FeaturesSelection
from steps.step03_feature_engineering import FeatureEngineering

def modelling_pipeline():
    
    print('\nModelling pipeline started')

    # Preprocessing
    print()
    print('Running preprocessing step...')
    PreProcessorStep().run()

    # Features Creation
    print()
    print('Running features creation step...') 
    FeaturesCreation().run()

    # Features Selection
    print()
    print('Running features selection step...') 
    FeaturesSelection().run()
    
    # Feature Engineering
    print()
    print('Running feature engineering step...')
    FeatureEngineering().run()

def main():
    modelling_pipeline()

if __name__ == '__main__':
    main()