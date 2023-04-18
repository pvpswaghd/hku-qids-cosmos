# Basic packages to import (NO NEED TO ADJUST IT)
import pandas as pd
    
#The addresses for the train data inputs (NO NEED TO ADJUST IT)
from config import Path
historical_data_path = Path.historical_data_path
MARKET_DATA_PATH = f'{historical_data_path}/second_round_train_market_data.csv'
FUNDAMENTAL_DATA_PATH = f'{historical_data_path}/second_round_train_fundamental_data.csv'
RETURN_DATA_PATH = f'{historical_data_path}/second_round_train_return_data.csv'

###################################################################################################################
# TODO: Import packages you will need, and realize any initialization you will need
import copy
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import random
#from sklearn.model_selection import TimeSeriesSplit
#from sklearn.metrics import mean_squared_error
import talib as ta
#import lightgbm as lgb

#silence warnings output...
import warnings
warnings.filterwarnings("ignore")

#following for the ML part...setting a seed meant that multiple executions will yield the same result.
random_state = 42
random.seed(random_state)
data_storage = dict()
model_storage = dict()
initialized = False
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

    #initialize data and model
    global initialized
    if not initialized:
        initialization()
        initialized = True
    features = build_features()

    predict_return_series = np.array([])
    df_test = add_features(merge_columns(market_df, fundamental_df))
    for a in range(54):
        df_a = df_test[df_test['asset'] == a]
        prediction = model_storage[a].predict(df_a[features])
        predict_return_series = np.concatenate((predict_return_series, prediction), axis=0)
    #OUTPUT
    predict_return_series = pd.Series(predict_return_series)

    threshold_value = 0.025
    decision_binary_series = predict_return_series.apply(lambda x: 1 if x > threshold_value else 0)
    equally_weighted_decision_series = decision_binary_series / decision_binary_series.sum()
    for each_weight in equally_weighted_decision_series:
        decision_list.append(each_weight)    
    
###################################################################################################################

    # Output the decision at this moment
    return decision_list

###################################################################################################################
# TODO: Write all functions that you will need below
def initialization():
    global data_storage
    global model_storage
    data_storage = dict()
    model_storage = dict()
    #read and load individual data to the storage
    df_train_market, df_train_fund, df_train_returns = load_historical_data()
    df = add_features(merge_columns(df_train_market, df_train_fund, df_train_returns))
    for a in range(54):
        data_storage[a] = df[df['asset'] == a]
    store_models()


#Manipulating Data
def interval_split(dt):
    #for date_time column of "sXXdXXpXX"
    f2, f3 = dt.find("d"), dt.find("p")
    return [int(dt[1:f2]), int(dt[f2+1:f3]), int(dt[f3+1:])]
def date_split(dt):
    #for date_time column of "sXXdXX"
    f2 = dt.find("d")
    return [int(dt[1:f2]), int(dt[f2+1:])]
def add_interval(df):
    df_interval_data = np.vstack(df.date_time.apply(lambda x: interval_split(x)))
    df[["asset", "day", "interval"]] = df_interval_data
    return df
def add_date(df):
    df_date_data = np.vstack(df.date_time.apply(lambda x: date_split(x)))
    df[["asset", "day"]] = df_date_data
    return df

#Features Generation
def add_period(df):
    #no intervals
    df_period = df[['asset', 'day']]
    for i in range(2, 15):
        df_period[f"period{i}"] = df["day"] // i
    return df_period

def add_remainder(df):
    #no intervals
    df_remainder = df[['asset', 'day']]
    for i in range(2, 15):
        df_remainder[f"remainder{i}"] = df["day"] % i
    return df_remainder

def ctc_returns(df_market_a):
    #no intervals
    df_ctc = df_market_a[['asset', 'day']].reset_index(drop=True)
    for days in [1, 5, 10, 20]:
        df_ctc[f"ctc{days}"] = df_market_a['close'].pct_change(days).reset_index(drop=True)
    return df_ctc

def daily_volume_moving_ratio(df_market_a):
    #contains intervals
    #take df by asset type
    daily_volume_series = df_market_a.groupby(df_market_a['day'])['volume'].sum()
    df_movv = df_market_a[['asset', 'day']].drop_duplicates(subset=["asset", "day"], keep='last').reset_index(drop=True)
    df_movv["daily_volume_moving_ratio_5d"] = daily_volume_series / daily_volume_series.rolling(5).mean()
    df_movv["daily_volume_moving_ratio_10d"] = daily_volume_series / daily_volume_series.rolling(10).mean()
    df_movv["daily_volume_moving_ratio_20d"] = daily_volume_series / daily_volume_series.rolling(20).mean()
    return df_movv

#Talib Features **
def add_talib_features(df_market_a):
    #This function take reference to the "construct_talib_features" function and tailor to our dataframe settings
    #by creating asset, day, interval columns (prior), following code should be more understandable
    close_p = df_market_a[df_market_a['interval'] == 50]['close'].reset_index(drop=True)
    high_p = df_market_a.groupby(df_market_a['day'])['high'].max().reset_index(drop=True)
    low_p = df_market_a.groupby(df_market_a['day'])['low'].min().reset_index(drop=True)
    feature_df = df_market_a[['asset', 'day']].drop_duplicates(subset=["asset", "day"], keep='last').reset_index(drop=True)
    feature_df["SAR"] = ta.SAR(high_p, low_p, acceleration = 0, maximum = 0)
    feature_df["SAREXT"] = ta.SAREXT(high_p, low_p, startvalue = 0, offsetonreverse = 0, accelerationinitlong = 0, accelerationlong = 0, accelerationmaxlong = 0, accelerationinitshort = 0, accelerationshort = 0, accelerationmaxshort = 0)
    feature_df["RSI"] = ta.RSI(close_p, timeperiod = 14) - 50
    feature_df["HT_DCPERIOD"] = ta.HT_DCPERIOD(close_p)
    feature_df["HT_PHASOR_inphase"], feature_df["HT_PHASOR_quadrature"] = ta.HT_PHASOR(close_p)
    feature_df["HT_SINE_sine"], feature_df["HT_SINE_leadsine"] = ta.HT_SINE(close_p)
    return feature_df

#implementing features
#Features in Consideration
def merge_columns(market, fundamental, returns=pd.DataFrame([0])):
    market = add_interval(market)
    fundamental = add_date(fundamental)
    df = pd.merge(market, fundamental, left_on=["asset", "day"], right_on=["asset", "day"])
    if returns.any()[0]:    #check if returns is a non-empty dataframe
        returns = add_date(returns)
        df = pd.merge(df, returns, left_on=["asset", "day"], right_on=["asset", "day"])
    # Note: This Step Will Mean that Metrics regarding
    return df
    # df = df.drop_duplicates(subset=["asset", "day"], keep='last').reset_index(drop=True)

def add_features(df):
    #THIS will return columns with only one set of ['asset', 'day'] (no intervals)
    df_close_only = df.drop_duplicates(subset=["asset", "day"], keep='last').reset_index(drop=True)
    df_features = df_close_only[['asset', 'day']]     #a df that contains the new feature to reduce loading time in each function call.

    #features not using intervals data
    df_features = pd.merge(df_features, add_period(df_close_only), left_on=["asset", "day"], right_on=["asset", "day"])
    df_features = pd.merge(df_features, add_remainder(df_close_only), left_on=["asset", "day"], right_on=["asset", "day"])
    df_features = pd.merge(df_close_only, df_features, left_on=["asset", "day"], right_on=["asset", "day"])

    #ctc returns
    ctc_features = pd.concat([ctc_returns(df_close_only[df_close_only['asset'] == i]) for i in range(54)]).reset_index(drop=True)
    df_features = pd.merge(df_features, ctc_features, left_on=["asset", "day"], right_on=["asset", "day"])

    #moving volume
    movv_features = pd.concat([daily_volume_moving_ratio(df[df['asset'] == i]) for i in range(54)]).reset_index(drop=True)
    df_features = pd.merge(df_features, movv_features, left_on=["asset", "day"], right_on=["asset", "day"])

    #talib features: iterate over each asset together, then merge by ["asset", "day"]
    talib_features = pd.concat([add_talib_features(df[df['asset'] == i]) for i in range(54)]).reset_index(drop=True)
    df_features = pd.merge(df_features, talib_features, left_on=["asset", "day"], right_on=["asset", "day"])
    return df_features

def build_features():
    #f for features (build the features by adding lists)
    f_base = ['turnoverRatio', 'transactionAmount', 'pe_ttm', 'pe', 'pb', 'ps', 'pcf']
    f_period = [f"period{i}" for i in range(2, 15)]
    f_remainder = [f"remainder{i}" for i in range(2, 15)]
    f_remainder_trimmed = ['remainder7', 'remainder14']
    f_ctc = ['ctc1', 'ctc5', 'ctc10', 'ctc20']
    f_movv = ['daily_volume_moving_ratio_5d', 'daily_volume_moving_ratio_10d', 'daily_volume_moving_ratio_20d']
    f_ta = ['SAR', 'SAREXT', 'RSI', 'HT_DCPERIOD', 'HT_PHASOR_inphase', 'HT_PHASOR_quadrature', 'HT_SINE_sine', 'HT_SINE_leadsine']
    f_lag = [f"lag{i}" for i in [700, 750, 800, 850, 900, 950]]

    return f_base + f_remainder_trimmed  + f_ctc + f_movv

def store_models():
    global model_storage
    features = build_features()
    target = 'return'
    for a in range(54):
        df_train = data_storage[a]
        x_all = df_train[features]
        y_all = df_train[target]

        reg = xgb.XGBRegressor(n_estimators=2000,
                booster="gbtree",
                objective="reg:linear",
                max_depth=2,            #high value leads to overfitting
                learning_rate=0.4,
                min_child_weight=6,             #higher value prevent overfitting (1000:700 ratio makes it easy to overfit)
                subsample=1,
                )
        
        reg.fit(x_all, y_all, eval_set=[(x_all, y_all)], verbose=50)
        model_storage[a] = reg
