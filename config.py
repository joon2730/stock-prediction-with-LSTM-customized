import os
import time
import numpy as np
import pandas as pd
import logging
import sys

class Config:
    # Data parameters
    feature_columns = list(range(2, 9))  # Feature columns (0-based indexing). Can also use custom list like [2,4,6,8]
    label_columns = [4, 5]               # Columns to predict (0-based indexing), e.g. 4th and 5th columns: low and high prices
    label_in_feature_index = (lambda x, y: [x.index(i) for i in y])(feature_columns, label_columns)  # Mapping labels in feature list

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
    debug_mode = False
    debug_num = 500  # Use only N rows in debug mode

    # Framework settings
    model_name = "model_" + continue_flag + ".pth"

    # Paths
    train_data_path = "./data/stock_data.csv"
    model_save_path = "./checkpoint/"
    figure_save_path = "./figure/"
    log_save_path = "./log/"
    do_log_print_to_screen = True
    do_log_save_to_file = True
    do_figure_save = False
    do_train_visualized = False  # Training loss visualization (visdom for PyTorch, tensorboardX for TF)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)
    if do_train and (do_log_save_to_file or do_train_visualized):
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        log_save_path = log_save_path + cur_time + '_' + "/"
        os.makedirs(log_save_path)
