3# Basic packages to import (NO NEED TO ADJUST IT)
import pandas as pd
    
# The addresses for the train data inputs (NO NEED TO ADJUST IT)
from config import Path
historical_data_path = Path.historical_data_path
MARKET_DATA_PATH = f'{historical_data_path}/second_round_train_market_data.csv'
FUNDAMENTAL_DATA_PATH = f'{historical_data_path}/second_round_train_fundamental_data.csv'
RETURN_DATA_PATH = f'{historical_data_path}/second_round_train_return_data.csv'

###################################################################################################################
# TODO: The following addresses are for your own test only; pls remove it before you submit your code
# MARKET_DATA_PATH = r"D:\HKU QIDS Affairs\historical_data\second_round_train_market_data.csv"
# FUNDAMENTAL_DATA_PATH = r"D:\HKU QIDS Affairs\historical_data\second_round_train_fundamental_data.csv"
# RETURN_DATA_PATH = r"D:\HKU QIDS Affairs\historical_data\second_round_train_return_data.csv"
###################################################################################################################

###################################################################################################################
# TODO: Import packages you will need, and realize any initialization you will need
import copy
import time
import numpy as np
import random
from sklearn import metrics
from sklearn.model_selection import train_test_split
import talib as ta
import lightgbm as lgb
from hyperopt import fmin, hp, partial, tpe, Trials

# GOOD LUCK!!!!!!!!!!
random_state = 42
random.seed(random_state)

# Set the global variables
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
    
    # Initialize the whole system
    global initialized
    if not initialized:
        initialization()
        initialized = True
    
    # Clarify the global variables for data and model storage
    global all_data_storage_object
    global model_storage_object
    
    # Get date information
    del market_df["date"]
    del fundamental_df["date"]
    new_data_date = int(str(market_df.date_time.map(lambda x: x.split("d")[1]).unique()[0])[:-2])
    print("Processing Data: d", new_data_date)
    
    # Add new data into the previous dataset dictionary
    market_df = market_df.set_index("date_time")
    fundamental_df = fundamental_df.set_index("date_time")
    for each_investment in all_data_storage_object:
        all_data_storage_object[each_investment]["market_data"] = all_data_storage_object[each_investment]["market_data"].append(market_df[market_df.index.map(lambda x: True if x.split("d")[0] == each_investment else False)])
        all_data_storage_object[each_investment]["fundamental_data"] = all_data_storage_object[each_investment]["fundamental_data"].append(fundamental_df[fundamental_df.index.map(lambda x: True if x.split("d")[0] == each_investment else False)])
    
    # Process X for test dataset, and process the extreme values
    all_X_dict = prepare_test_dataset()
    for each_feature in model_storage_object["extreme_value_dict"]:
        for each_investment in model_storage_object["extreme_value_dict"][each_feature]:
            lower_quantile_value, upper_quantile_value = model_storage_object["extreme_value_dict"][each_feature][each_investment][0], model_storage_object["extreme_value_dict"][each_feature][each_investment][1]
            all_X_dict[each_investment][each_feature] = all_X_dict[each_investment][each_feature].apply(lambda x: upper_quantile_value if x > upper_quantile_value else x)
            all_X_dict[each_investment][each_feature] = all_X_dict[each_investment][each_feature].apply(lambda x: lower_quantile_value if x < lower_quantile_value else x)
    
    # Make prediction for each investment
    predicted_value_series = pd.Series(index = ["s" + str(i) for i in range(54)])
    for each_investment in all_X_dict:
        selected_X = copy.deepcopy(all_X_dict[each_investment])
        selected_X["investment_code"] = selected_X.index.map(lambda x: int(x.split("d")[0][1:]))
        selected_X = selected_X.reset_index(drop = True)
        selected_X["investment_code"] = selected_X["investment_code"].astype("category")
        if selected_X.dropna().shape[0] != 0:
            predicted_value_series[each_investment] = model_storage_object["model"].predict(selected_X)[0]
    
    # Make decisions and equally weight each decision
    threshold_value = 0.01
    decision_binary_series = predicted_value_series.apply(lambda x: 1 if x > threshold_value else 0)
    equally_weighted_decision_series = decision_binary_series / decision_binary_series.sum()
    for each_weight in equally_weighted_decision_series:
        decision_list.append(each_weight)

###################################################################################################################

    # Output the decision at this moment
    return decision_list

###################################################################################################################
# TODO: Write all functions that you will need below

#%%

# Working functions

# Initialization: Load historical data and train the first model
def initialization():
    print('Start Initializaion...')

    # Clarify the global variables for data and model storage
    global all_data_storage_object
    global model_storage_object
    
    # Load historical data
    all_data_dict = {}
    market_data_df, fundamental_data_df, return_data_df = load_historical_data()
    for each_investment_code in range(54):
        complete_investment_code_str = "s" + str(each_investment_code)
        all_data_dict[complete_investment_code_str] = {}
        all_data_dict[complete_investment_code_str]["market_data"] = market_data_df[market_data_df.date_time.apply(lambda x: True if x.split("d")[0] == complete_investment_code_str else False)].set_index("date_time")
        all_data_dict[complete_investment_code_str]["fundamental_data"] = fundamental_data_df[fundamental_data_df.date_time.apply(lambda x: True if x.split("d")[0] == complete_investment_code_str else False)].set_index("date_time")
        all_data_dict[complete_investment_code_str]["return_data"] = return_data_df[return_data_df.date_time.apply(lambda x: True if x.split("d")[0] == complete_investment_code_str else False)].set_index("date_time")

    # Adjust the global variable for data storage
    all_data_storage_object = all_data_dict

    # Train model
    model, extreme_value_dict = train_model()
    model_dict = {}
    model_dict["model"] = model
    model_dict["extreme_value_dict"] = extreme_value_dict

    # Adjust the global variable for model storage
    model_storage_object = model_dict
    
    # Progress report
    print("Initialization Finish!")

# Prepare test dataset
def prepare_test_dataset():
    
    # Clarify the global variable for data storage
    global all_data_storage_object
    
    # Process each investment
    all_X_dict = {}
    for each_investment in all_data_storage_object:
        
        # Construct X and store the result
        market_data_df = all_data_storage_object[each_investment]["market_data"]
        fundamental_data_df = all_data_storage_object[each_investment]["fundamental_data"]
        all_features = construct_all_features(market_data_df, fundamental_data_df)
        all_X_dict[each_investment] = all_features.iloc[[-1], :]
    
    # Return
    return all_X_dict

# Construct all features
def construct_all_features(market_data_df, fundamental_data_df):
    
    # Store all features
    all_feature_list = []
    
    # Call every feature function
    ctc_returns = construct_ctc_returns(market_data_df)
    daily_volume_moving_ratios = construct_daily_volume_moving_ratios(market_data_df)
    morning_and_tail_volume_moving_ratios = construct_open_and_close_volume_moving_ratios(market_data_df)
    period_and_overall_turnover_ratios = construct_period_and_overall_turnover_ratios(market_data_df, fundamental_data_df)
    all_feature_list.append(ctc_returns)
    all_feature_list.append(daily_volume_moving_ratios)
    all_feature_list.append(morning_and_tail_volume_moving_ratios)
    all_feature_list.append(period_and_overall_turnover_ratios)
    talib_features = construct_talib_features(market_data_df)
    all_feature_list.append(talib_features)

    # Combine all features
    all_features = pd.concat(all_feature_list, axis = 1)

    # Return
    return all_features

# Construct the latest y
def construct_Y(market_data_df):
    
    # Find the correct timeplot
    selected_data_df = market_data_df[market_data_df.index.map(lambda x: True if x.split("p")[1] == "50" else False)]
    selected_data_df.index = selected_data_df.index.map(lambda x: x.split("p")[0])
    
    # Start the calculation
    y_to_predict_series = selected_data_df.close.pct_change(2).shift(-2)
    y_to_predict_series.name = "y_to_predict"

    # Return
    return y_to_predict_series

# Process the train dataset
def prepare_train_dataset(train_length):
    
    # Clarify the global variable for data storage
    global all_data_storage_object
    
    # Process each investment
    all_y_and_X_dict = {}
    for each_investment in all_data_storage_object:
        
        # Construct y and X
        market_data_df = all_data_storage_object[each_investment]["market_data"]
        fundamental_data_df = all_data_storage_object[each_investment]["fundamental_data"]
        all_features = construct_all_features(market_data_df, fundamental_data_df)
        y_to_predict = construct_Y(market_data_df)
        
        # Combine y and X
        y_and_X_df = pd.concat([y_to_predict, all_features], axis = 1)
        y_and_X_df = y_and_X_df[-train_length:]
        all_y_and_X_dict[each_investment] = y_and_X_df
    
    # Return
    return all_y_and_X_dict

# Train model
def train_model():
    
    # Basic parameter for rolling-window testing
    train_length = 1500

    # Prepare the train dataset
    all_y_and_X_dict = prepare_train_dataset(train_length)
    
    # Process data for each investment
    train_df_component_list = []
    extreme_value_dict = {}
    for each_investment in all_y_and_X_dict:
        temp_df = copy.deepcopy(all_y_and_X_dict[each_investment])
        temp_df["investment_code"] = each_investment
        temp_df["date"] = temp_df.index.map(lambda x: int(x.split("d")[1]))
        temp_df = temp_df.set_index(["investment_code", "date"])

        # Remove abnormal values
        temp_df = temp_df.replace(np.inf, np.nan)
        temp_df = temp_df.dropna()
        train_df_component_list.append(temp_df)
    
        # Process extreme values
        for each_feature in temp_df.columns[1:]:
            if each_feature in ["close_price"]:
                continue
            lower_quantile_value, upper_quantile_value = temp_df[each_feature].quantile(0.005), temp_df[each_feature].quantile(0.995)
            temp_df[each_feature] = temp_df[each_feature].apply(lambda x: upper_quantile_value if x > upper_quantile_value else x)
            temp_df[each_feature] = temp_df[each_feature].apply(lambda x: lower_quantile_value if x < lower_quantile_value else x)
            if each_feature not in extreme_value_dict:
                extreme_value_dict[each_feature] = {}
            extreme_value_dict[each_feature][each_investment] = [lower_quantile_value, upper_quantile_value]
            
    # Combine all datasets
    train_df = pd.concat(train_df_component_list)
        
    # Process extreme values for y
    lower_quantile_value, upper_quantile_value = train_df["y_to_predict"].quantile(0.0002), train_df["y_to_predict"].quantile(0.9998)
    train_df["y_to_predict"] = train_df["y_to_predict"].apply(lambda x: np.nan if x < lower_quantile_value or x > upper_quantile_value else x)
    train_df = train_df.dropna()
    
    # Construct train sets and validation sets
    model_train_df = pd.DataFrame()
    model_validation_df = pd.DataFrame()
    model_complete_df = train_df.reset_index()
    del model_complete_df["date"]
    model_complete_df["investment_code"] = model_complete_df["investment_code"].map(lambda x: int(x.split("s")[1]))
    model_complete_df["investment_code"] = model_complete_df["investment_code"].astype("category")
    for each_investment in model_complete_df["investment_code"].unique():
        temp_selected_df = model_complete_df[model_complete_df.investment_code == each_investment]
        sample_train_df = temp_selected_df.sample(frac = 0.8)
        model_train_df = model_train_df.append(sample_train_df, ignore_index = True)
        model_validation_df = model_validation_df.append(temp_selected_df.drop(sample_train_df.index), ignore_index = True)
    
    # Generate LightGBM data type
    train_X = model_train_df.drop(columns = ["y_to_predict"])
    train_Y = model_train_df["y_to_predict"]
    validation_X = model_validation_df.drop(columns = ["y_to_predict"])
    validation_Y = model_validation_df["y_to_predict"]
    train_data_lgb = lgb.Dataset(data = train_X, label = train_Y, categorical_feature = ["investment_code"])
    validation_data_lgb = lgb.Dataset(data = validation_X, label = validation_Y, categorical_feature = ["investment_code"])

    def lgb_argsDict_tranform(argsDict):
        argsDict["max_depth"] = argsDict["max_depth"] + 5
        argsDict["num_trees"] = argsDict["num_trees"] + 50
        argsDict["learning_rate"] = argsDict["learning_rate"] * 0.02 + 0.05
        argsDict["bagging_fraction"] = argsDict["bagging_fraction"] * 0.1 + 0.5
        argsDict["num_leaves"] = argsDict["num_leaves"] * 3 + 10
        argsDict["bagging_freq"] = argsDict["num_leaves"] + 10
        argsDict["subsample"] = argsDict["subsample"] * 0.02 + 0.05
        return argsDict
    
    def mape(y_test, y_predicted):
        y_test, y_predicted = np.array(y_test), np.array(y_predicted)
        temp = pd.Series((y_test - y_predicted) / y_test)
        mape = np.mean(np.abs(temp.replace(np.inf, 0).replace(-np.inf, 0)))
        return mape
    
    def lgb_get_model_score_mape(model):
        prediction = model.predict(validation_X, num_iteration = model.best_iteration)
        y_predicted = pd.Series(prediction)
        return mape(validation_Y, y_predicted)
    
    def lightgbm_factory(argsDict):
        argsDict = lgb_argsDict_tranform(argsDict)
        params = {"nthread": -1,  
                  "max_depth": argsDict["max_depth"],
                  "num_trees": argsDict["num_trees"],
                  "eta": argsDict["learning_rate"],
                  "bagging_fraction": argsDict["bagging_fraction"],
                  "num_leaves": argsDict["num_leaves"],
                  "bagging_freq": argsDict["bagging_freq"],
                  "subsample": argsDict["subsample"],
                  "objective": "regression",
                  "feature_fraction": 0.7,
                  "lambda_l1": 0,
                  "lambda_l2": 0,
                  "bagging_seed": random_state
                 }
        params["metric"] = ["mape"]
        model_lgb = lgb.train(params = params, train_set = train_data_lgb, num_boost_round = 300, valid_sets = [validation_data_lgb], early_stopping_rounds = 100, categorical_feature = ["investment_code"])
        return lgb_get_model_score_mape(model_lgb)

    # Find a best model
    lgb_space = {"max_depth": hp.randint("max_depth", 15),
                 "num_trees": hp.randint("num_trees", 10),
                 "learning_rate": hp.uniform("learning_rate", 0.001, 0.5),
                 "bagging_fraction": hp.randint("bagging_fraction", 5),
                 "num_leaves": hp.randint("num_leaves", 10),
                 "bagging_freq": hp.randint("bagging_freq", 1),
                 "subsample": hp.uniform("subsample", 0.1, 0.5)
                }
    algo = partial(tpe.suggest, n_startup_jobs = 1)
    lgb_best = fmin(fn = lightgbm_factory, space = lgb_space, algo = algo, max_evals = 20, pass_expr_memo_ctrl = None)
    opt_res = lgb_argsDict_tranform(lgb_best)
    model = lgb.train(params = opt_res, train_set = train_data_lgb, num_boost_round = 20, valid_sets = [train_data_lgb, validation_data_lgb], categorical_feature = ["investment_code"])

    # Return
    return model, extreme_value_dict

#%%

# FEATURE LIBRARY!!!

def construct_ctc_returns(market_data_df):
    selected_data_df = market_data_df[market_data_df.index.map(lambda x: True if x.split("p")[1] == "50" else False)]
    selected_data_df.index = selected_data_df.index.map(lambda x: x.split("p")[0])
    result_df = pd.DataFrame()
    result_df["close_price"] = selected_data_df.close
    result_df["ctc_return_1d"] = selected_data_df.close.pct_change(1)
    result_df["ctc_return_5d"] = selected_data_df.close.pct_change(5)
    result_df["ctc_return_10d"] = selected_data_df.close.pct_change(10)
    result_df["ctc_return_20d"] = selected_data_df.close.pct_change(20)
    return result_df

def construct_daily_volume_moving_ratios(market_data_df):
    daily_volume_series = market_data_df.groupby(market_data_df.index.map(lambda x: x.split("p")[0]), sort = False).volume.sum()
    result_df = pd.DataFrame()
    result_df["daily_volume_moving_ratio_5d"] = daily_volume_series / daily_volume_series.rolling(5).mean()
    result_df["daily_volume_moving_ratio_10d"] = daily_volume_series / daily_volume_series.rolling(10).mean()
    result_df["daily_volume_moving_ratio_20d"] = daily_volume_series / daily_volume_series.rolling(20).mean()
    return result_df

def construct_open_and_close_volume_moving_ratios(market_data_df):
    morning_session_selected_data = market_data_df[market_data_df.index.map(lambda x: True if x.split("p")[1] in ["1", "2"] else False)]
    tail_session_selected_data = market_data_df[market_data_df.index.map(lambda x: True if x.split("p")[1] in ["49", "50"] else False)]
    morning_session_volume_series = morning_session_selected_data.groupby(morning_session_selected_data.index.map(lambda x: x.split("p")[0]), sort = False).volume.sum()
    tail_session_volume_series = tail_session_selected_data.groupby(tail_session_selected_data.index.map(lambda x: x.split("p")[0]), sort = False).volume.sum()
    result_df = pd.DataFrame()
    result_df["morning_session_volume_moving_ratio_5d"] = morning_session_volume_series / morning_session_volume_series.rolling(5).mean()
    result_df["tail_session_volume_moving_ratio_5d"] = tail_session_volume_series / tail_session_volume_series.rolling(5).mean()
    return result_df

def construct_period_and_overall_turnover_ratios(market_data_df, fundamental_data_df):
    result_df = pd.DataFrame()
    result_df["turnoverRatio"] = fundamental_data_df["turnoverRatio"]
    result_df["transactionAmount"] = fundamental_data_df["transactionAmount"]
    result_df["pe_ttm"] = fundamental_data_df["pe_ttm"]
    result_df["pb"] = fundamental_data_df["pb"]
    result_df["ps"] = fundamental_data_df["ps"]
    result_df["pcf"] = fundamental_data_df["pcf"]
    morning_session_selected_data = market_data_df[market_data_df.index.map(lambda x: True if x.split("p")[1] in ["1", "2"] else False)]
    tail_session_selected_data = market_data_df[market_data_df.index.map(lambda x: True if x.split("p")[1] in ["49", "50"] else False)]
    morning_session_volume_series = morning_session_selected_data.groupby(morning_session_selected_data.index.map(lambda x: x.split("p")[0]), sort = False).volume.sum()
    tail_session_volume_series = tail_session_selected_data.groupby(tail_session_selected_data.index.map(lambda x: x.split("p")[0]), sort = False).volume.sum()
    daily_volume_series = market_data_df.groupby(market_data_df.index.map(lambda x: x.split("p")[0]), sort = False).volume.sum()
    result_df["morning_session_turnoverRatio"] = fundamental_data_df["turnoverRatio"] * morning_session_volume_series / daily_volume_series
    result_df["tail_session_turnoverRatio"] = fundamental_data_df["turnoverRatio"] * tail_session_volume_series / daily_volume_series
    return result_df

def construct_talib_features(market_data_df):
    open_p = market_data_df.groupby(market_data_df.index.map(lambda x: x.split("p")[0]), sort = False).open.head(1)
    open_p.index = open_p.index.map(lambda x: x.split("p")[0])
    close_p = market_data_df.groupby(market_data_df.index.map(lambda x: x.split("p")[0]), sort = False).close.tail(1)
    close_p.index = close_p.index.map(lambda x: x.split("p")[0])
    high_p = market_data_df.groupby(market_data_df.index.map(lambda x: x.split("p")[0]), sort = False).high.max()
    low_p = market_data_df.groupby(market_data_df.index.map(lambda x: x.split("p")[0]), sort = False).low.min()
    feature_df = pd.DataFrame([])
    feature_df["SAR"] = ta.SAR(high_p, low_p, acceleration = 0, maximum = 0)
    feature_df["SAREXT"] = ta.SAREXT(high_p, low_p, startvalue = 0, offsetonreverse = 0, accelerationinitlong = 0, accelerationlong = 0, accelerationmaxlong = 0, accelerationinitshort = 0, accelerationshort = 0, accelerationmaxshort = 0)
    feature_df["RSI"] = ta.RSI(close_p, timeperiod = 14) - 50
    feature_df["HT_DCPERIOD"] = ta.HT_DCPERIOD(close_p)
    feature_df["HT_PHASOR_inphase"], feature_df["HT_PHASOR_quadrature"] = ta.HT_PHASOR(close_p)
    feature_df["HT_SINE_sine"], feature_df["HT_SINE_leadsine"] = ta.HT_SINE(close_p)
    return feature_df
