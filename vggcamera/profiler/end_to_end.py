"""
end_to_end.py
"""
import os
import argparse
import yaml
import logging
import logging.config

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D


def establish_logging(verbosity=0):
    """
    Establish logging for the app.
    """
    # Attempt to load the config file; otherwise give warning and default to standard
    LOGGING_CONFIG="logging.yaml"  # It'd be nice to have these as module-wide constants.
    if verbosity >= 1:
        LOGGING_CONFIG="logging.verbose.yaml"
    if os.path.exists(LOGGING_CONFIG):
        with open(LOGGING_CONFIG, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


def main(model=""):
    """
    Build a Keras model, and run a through images through it.
    :param model:
    :return:
    """



if __name__ == "__main__":

    # Parse arguments
    PROGRAM_DESCRIPTION = "Capture and classify an image."
    parser = argparse.ArgumentParser(PROGRAM_DESCRIPTION)
    parser.add_argument('-m', '--model', help="What Keras model to run", default="")
    args = parser.parse_args()

    # Establish logging
    logger = establish_logging(verbosity=args.verbosity)

    # Entry point goes here
    main(model=args.model)