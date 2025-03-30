import pandas as pd
import numpy as np
import os
import sys
import time
import logging

from utils import load_logger, draw
from data import Data
from model import train, predict
from config import Config

def main(config):
    logger = load_logger(config)
    try:
        np.random.seed(config.random_seed)
        data_gainer = Data(config)

        if config.do_train:
            train_X, valid_X, train_Y, valid_Y = data_gainer.get_train_and_valid_data()
            train(config, logger, [train_X, train_Y, valid_X, valid_Y])

        if config.do_predict:
            test_X, test_Y = data_gainer.get_test_data(return_label_data=True)
            pred_result = predict(config, test_X)
            draw(config, data_gainer, logger, pred_result)
    except Exception:
        logger.error("Run Error", exc_info=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # You can add command line arguments here if needed
    args = parser.parse_args()

    con = Config()
    for key in dir(args):
        if not key.startswith("_"):
            setattr(con, key, getattr(args, key))

    main(con)