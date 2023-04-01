# Basic packages to import (NO NEED TO ADJUST IT)
import pandas as pd
    
# The addresses for the train data inputs (NO NEED TO ADJUST IT)
from config import Path
historical_data_path = Path.historical_data_path
MARKET_DATA_PATH = f'{historical_data_path}/second_round_train_market_data.csv'
FUNDAMENTAL_DATA_PATH = f'{historical_data_path}/second_round_train_fundamental_data.csv'
RETURN_DATA_PATH = f'{historical_data_path}/second_round_train_return_data.csv'

###################################################################################################################
# TODO: The following addresses are for your own test only; pls remove it before you submit your code
MARKET_DATA_PATH = ""
FUNDAMENTAL_DATA_PATH = ""
RETURN_DATA_PATH = ""
###################################################################################################################

###################################################################################################################
# TODO: Import packages you will need, and realize any initialization you will need

###################################################################################################################

# Help load the historical data (NO NEED TO ADJUST IT)
def load_historical_data():
    market_data = pd.read_csv(MARKET_DATA_PATH)
    fundamental_data = pd.read_csv(FUNDAMENTAL_DATA_PATH)
    return_data = pd.read_csv(RETURN_DATA_PATH)
    return market_data, fundamental_data, return_data

# Important: Get your decision output
def get_decisions(market_df: pd.DataFrame, fundamental_df: pd.DataFrame):

    # Store the decisions here
    decision_list = []

###################################################################################################################
    # TODO: Write your code here
    
###################################################################################################################

    # Output the decision at this moment
    return decision_list

###################################################################################################################
# TODO: Write all functions that you will need below
