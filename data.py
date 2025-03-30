import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import Config
from rawdata import read_ohlcv_from_csv, fetch_ohlcv_online, append_additional_features

class Data:
    def __init__(self, config):
        self.config = config
        self.data, self.data_column_name = self.read_data()

        self.label_in_feature_index = (lambda x, y: [x.index(i) for i in y]) \
            (self.data_column_name, config.label_columns)  # Mapping labels in feature list
        
        self.data_num = self.data.shape[0]
        self.train_num = int(self.data_num * self.config.train_data_rate)

        self.mean = np.mean(self.data, axis=0)
        self.std = np.std(self.data, axis=0)
        self.norm_data = (self.data - self.mean) / self.std  # Normalize data

        self.start_num_in_test = 0  # First few days in test set will be dropped

    # def read_data(self):
    #     if self.config.debug_mode:
    #         init_data = pd.read_csv(self.config.train_data_path, nrows=self.config.debug_num,
    #                                 usecols=self.config.feature_columns)
    #     else:
    #         init_data = pd.read_csv(self.config.train_data_path, usecols=self.config.feature_columns)
    #     return init_data.values, init_data.columns.tolist()

    # customized read_data function for custom stock data
    def read_data(self):
        if self.config.fetch_data_online:
            ohlcv = fetch_ohlcv_online(
                ticker=self.config.ticker,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                interval=self.config.interval,
            )
        else:
            ohlcv = read_ohlcv_from_csv(self.config)

        init_data = append_additional_features(ohlcv, to_append=self.config.add_features)
        return init_data.values, init_data.columns.tolist()

    def get_train_and_valid_data(self):
        feature_data = self.norm_data[:self.train_num]
        label_data = self.norm_data[self.config.predict_day : self.config.predict_day + self.train_num,
                                    self.label_in_feature_index]

        if not self.config.do_continue_train:
            # Non-continuous mode: every time_step rows as one sample, shifted by 1 row
            train_x = [feature_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
            train_y = [label_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
        else:
            # Continuous mode: every time_step rows as one sample, shifted by time_step rows
            train_x = [feature_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
                       for start_index in range(self.config.time_step)
                       for i in range((self.train_num - start_index) // self.config.time_step)]
            train_y = [label_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
                       for start_index in range(self.config.time_step)
                       for i in range((self.train_num - start_index) // self.config.time_step)]

        train_x, train_y = np.array(train_x), np.array(train_y)

        train_x, valid_x, train_y, valid_y = train_test_split(
            train_x, train_y,
            test_size=self.config.valid_data_rate,
            random_state=self.config.random_seed,
            shuffle=self.config.shuffle_train_data
        )
        return train_x, valid_x, train_y, valid_y

    def get_test_data(self, return_label_data=False):
        feature_data = self.norm_data[self.train_num:]
        sample_interval = min(feature_data.shape[0], self.config.time_step)
        self.start_num_in_test = feature_data.shape[0] % sample_interval
        time_step_size = feature_data.shape[0] // sample_interval

        test_x = [feature_data[self.start_num_in_test+i*sample_interval : self.start_num_in_test+(i+1)*sample_interval]
                  for i in range(time_step_size)]
        if return_label_data:
            label_data = self.norm_data[self.train_num + self.start_num_in_test:, self.label_in_feature_index]
            return np.array(test_x), label_data
        return np.array(test_x)