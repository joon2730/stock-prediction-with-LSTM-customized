import os
import time
import numpy as np
import pandas as pd
import logging
import sys
from datetime import datetime

class Config:
    # Data parameters
    fetch_data_online = False  # Set to True if reading from CSV

    if fetch_data_online:
        # read data from Yahoo Finance
        ticker = "AAPL"                  # Stock ticker symbol
        start_date = "2023-04-01"        # Start date for fetching data
        end_date = time.strftime("%Y-%m-%d", time.localtime())  # End date for fetching data
        interval = "1h"                 # Data interval (e.g., 1d, 1h, 5m, etc.)
    else:
        train_data_path = "data/BTCUSD/BTCUSD_1h_Binance.csv"  # Path to the CSV file
        # train_data_path = "data/stock_data.csv"  # Path to the CSV file

    add_features = {  # Additional features to append
        'log_return': [1],  # 1 for daily log return
        # 'sma': [20, 50],    # 20 and 50 days simple moving average
        # 'rsi': [14],        # 14 days relative strength index
    }

    base_features = ["Close", "Open", "High", "Low", "Volume"]  # Features to use for training
    derived_features = [f"{k}_{i}" for k, v in add_features.items() for i in v]

    feature_columns = base_features + derived_features  # All features to use

    label_columns = ["Low", 'High']  # Columns to predict

    predict_day = 1  # Number of days to predict into the future

    # Network parameters
    input_size = len(feature_columns)
    output_size = len(label_columns)

    hidden_size = 128  # LSTM hidden layer size
    lstm_layers = 2    # Number of stacked LSTM layers
    dropout_rate = 0.2
    time_step = 20     # How many days of data used to predict (also LSTM time steps). Make sure training data > time_step

    # Training parameters
    do_train = True
    do_predict = True
    add_train = False           # Load existing model for incremental training
    shuffle_train_data = True   # Shuffle training data
    use_cuda = False            # Use GPU training

    train_data_rate = 0.95
    valid_data_rate = 0.15

    batch_size = 64
    learning_rate = 0.001
    epoch = 20
    patience = 5                # Early stopping patience
    random_seed = 42

    do_continue_train = False   # Use final_state of last sample as init_state of next (only for RNNs in PyTorch)
    continue_flag = ""
    if do_continue_train:
        shuffle_train_data = False
        batch_size = 1
        continue_flag = "continue_"

    # Debug mode
    # debug_mode = False
    # debug_num = 500  # Use only N rows in debug mode

    # Framework settings
    model_name = "model_" + continue_flag + ".pth"

    # Paths
    model_save_path = "./checkpoint/"
    figure_save_path = "./figure/"
    log_save_path = "./log/"
    do_log_print_to_screen = True
    do_log_save_to_file = True
    do_figure_save = True
    do_train_visualized = False  # Training loss visualization (visdom for PyTorch)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)
    if do_train and (do_log_save_to_file or do_train_visualized):
        cur_time = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())
        if fetch_data_online:
            log_save_filename = ticker + '_' + interval + '_' + \
                datetime.strptime(start_date, "%Y-%m-%d").strftime("%y-%m-%d") + '_to_' + \
                datetime.strptime(end_date, "%Y-%m-%d").strftime("%y-%m-%d") + '.log'
        else:
            log_save_filename = train_data_path.split(".")[0].split("/")[-1] + ".log"
        log_save_path = log_save_path + cur_time + '_' + "/"
        os.makedirs(log_save_path)
