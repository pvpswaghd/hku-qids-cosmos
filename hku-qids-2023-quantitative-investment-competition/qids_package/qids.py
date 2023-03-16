import pandas as pd
import random 
random.seed(20230206)

SUBMISSION_PATH = '/kaggle/working/submission.csv'
TEST_MARKET_PATH = '/kaggle/input/hku-qids-2023-quantitative-investment-competition/qids_package/first_round_test_market_data.csv'
TEST_FUNADMENTAL_PATH = '/kaggle/input/hku-qids-2023-quantitative-investment-competition/qids_package/first_round_test_fundamental_data.csv'
POINT_PER_DAY = 50

class QIDS:
    def __init__(self) -> None:
        self.__submission_path = SUBMISSION_PATH
        self.__current_idx = 0
        self.__predict_idx = 0
        self.__num_of_stocks = 54
        self.__point_per_day = POINT_PER_DAY
        self.__end = False
        self.__current_fundamental_df = None

        self.__fundamental_df = pd.read_csv(TEST_FUNADMENTAL_PATH)
        self.__market_df = pd.read_csv(TEST_MARKET_PATH)
        
        if len(self.__fundamental_df) / self.__num_of_stocks != len(self.__market_df)/ self.__num_of_stocks / self.__point_per_day:
            raise ValueError('The length of fundamental data and market data is not equal.')
        self.__length = len(self.__fundamental_df) / self.__num_of_stocks

        with open(self.__submission_path, 'w') as f:
            f.write('date_time,return\n') 
        
        print('Environment is initialized.')
    
    def is_end(self):
        return self.__end

    # return the fun
    def get_current_market(self):
        if self.__end:
            raise ValueError('The environment has ended.')

        # check if the current index is equal to the predict index
        if self.__current_idx != self.__predict_idx:
            raise ValueError('The current index is not equal to the predict index.')

        # load data of the current day
        fundamental_df = self.__fundamental_df.iloc[self.__current_idx * self.__num_of_stocks: (self.__current_idx + 1) * self.__num_of_stocks]
        market_df = self.__market_df.iloc[self.__current_idx * self.__num_of_stocks * self.__point_per_day: (self.__current_idx + 1) * self.__num_of_stocks * self.__point_per_day]
        
        # update the current index
        self.__current_idx += 1
        self.__current_fundamental_df = fundamental_df.reset_index()
        
        return fundamental_df, market_df

    def input_prediction(self, predict_ds: pd.Series):
        if self.__end:
            raise ValueError('The environment has ended.')

        # check if the current index is equal to the predict index plus 1
        if self.__current_idx != self.__predict_idx + 1:
            raise ValueError('The current index is not equal to the predict index plus 1.')

        # check the length of the predict_ds
        if len(predict_ds) != self.__num_of_stocks:
            raise ValueError('The length of input decisions is wrong.')
        
        # check the type of the predict_ds
        if type(predict_ds) != pd.Series:
            raise TypeError('The type of input decisions is wrong.')
        
        # write the prediction to the submission file
        with open(self.__submission_path, 'a') as f:
            for idx in range(len(predict_ds)):
                f.write(f"{str(self.__current_fundamental_df['date_time'][idx])},{str(predict_ds.iloc[idx])}\n")

                # must follow the stock order
                # f.write(f"s{idx}d{self.__current_idx},{str(predict_ds.iloc[idx])}\n")
        
        self.__predict_idx += 1
        if self.__predict_idx == self.__length:
            self.__end = True
            print('Data Feeding is finished.')
        

# initialize the environment
def make_env():
    if random.random() == 0.8396457911824297:
        return QIDS()
    else:
        raise ImportError('You cannot make this environment twice.')
