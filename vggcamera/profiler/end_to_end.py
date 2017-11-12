"""
end_to_end.py
"""
import os
import argparse
import yaml
import logging
import logging.config

from time import time
from sys import exit

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

SINGLE_IMAGE_PATH   = "test_single"
MULTIPLE_IMAGE_PATH = "test_multiple"


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


def main(model_structure=None, single_runs=5):
    """
    Build a Keras model, and run a few images through it.
    :param model_structure: Which model structure to generate
    :return: n/a
    """
    logging.info("Specified model structure: {}".format(model_structure))
    t = time()
    if model_structure == "vgg":
        logging.info("Generating VGG Model.")
        model = create_vgg()
    if model_structure == "vgg_micro":
        logging.info("Generating VGG Micro Model.")
        model = create_vgg_micro()
    if model_structure == "vgg_nano":
        logging.info("Generating VGG Nano Model.")
        model = create_vgg_nano()
    else:
        logger.info("No/invalid model structure identified. Exiting program.")
        exit()
    t_model_assembly = time() - t

    logger.info("Compiling model...")
    t = time()
    model.compile(
        optimizer=Adam(lr=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    t_compilation = time() - t
    generator = ImageDataGenerator()
    logger.info("Done.")

    logger.info("Test 1: Run repeated analysis on a single image.")
    flow = generator.flow_from_directory(
        SINGLE_IMAGE_PATH,
        target_size=(224,224),
        class_mode='categorical',
        batch_size=1
    )
    t_single = []
    for i in range(single_runs):
        logger.info("Predicting...")
        t = time()
        predictions = model.predict_generator(generator=flow, val_samples=1)
        t_single.append(time() - t)
        logger.info("Done. Most likely outcome was #{}".format(predictions.argmax()))

    logger.info("Test 2: Run analysis on three images.")
    flow = generator.flow_from_directory(
        MULTIPLE_IMAGE_PATH,
        target_size=(224,224),
        class_mode='categorical',
        batch_size=1
    )
    logger.info("Predicting...")
    t = time()
    predictions = model.predict_generator(generator=flow, val_samples=3)
    t_multiple = time() - t
    logger.info("Done. Most likely outcome was #{}".format(predictions.argmax()))

    logger.info("===============")
    logger.info("TESTING RESULTS")
    logger.info("===============")
    logger.info("Specified model:     {0}".format(model_structure))
    logger.info("Model assembly time: {0}".format(t_model_assembly))
    logger.info("Compilation time:    {0}".format(t_compilation))
    for i in range(single_runs):
        logger.info("Single run {0}:        {1}".format(i, t_single[i]))
    logger.info("Multiple images:     {0}".format(t_multiple))


def create_vgg():
    """
    Create a garbage Keras model with the overall structure of the VGG16 network.
    :return: Keras model
    """
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1000, activation='softmax'))

    return model


def create_vgg_micro():
    """
    Create a garbage miniature Keras with a smaller convnet structure.
    :return: Keras model
    """
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1000, activation='softmax'))

    return model


def create_vgg_nano():
    """
    Create a garbage miniature, minuiature Keras model with a smaller convnet structure.
    :return: Keras model
    """
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(Flatten())

    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1000, activation='softmax'))

    return model


if __name__ == "__main__":

    # Parse arguments
    PROGRAM_DESCRIPTION = "Capture and classify an image."
    parser = argparse.ArgumentParser(PROGRAM_DESCRIPTION)
    parser.add_argument('-v', '--verbosity', action="count", help="Enable verbose output to file", default=0)
    parser.add_argument('-m', '--model', help="What Keras model to run", default="")
    args = parser.parse_args()

    # Establish logging
    logger = establish_logging(verbosity=args.verbosity)

    # Entry point goes here
    main(model_structure=args.model)