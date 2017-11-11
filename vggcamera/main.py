"""
main.py 
"""
import os
import argparse
import yaml
import logging
import logging.config

from os import path, chdir, makedirs 
from shutil import rmtree, copy

import picamera
from time import sleep, gmtime, strftime

import json


# Establish defaults
TEMP_ROOT = "/tmp"
TEMP_DIRECTORY = "test/unknown/"
TEMP_FILE = "output.bmp"
ARCHIVE_DIRECTORY = "/home/goyder/vgg_outputs"

def establish_logging(verbosity=0):
    """
    Establish logging for the app.
    """
    # Attempt to load the config file; otherwise give warning and default to standard
    LOGGING_CONFIG="logging.yaml"
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


def main(classify=True):
    """
    Main entry point for program.
    """
    logger.debug("Function is going to classify: {}".format(classify))
    if classify:
        from vgg import vgg16 
        from keras.utils import get_file
        from keras.preprocessing.image import ImageDataGenerator
        from numpy import save

    # Generate file paths
    logger.info("Generating temp filepath for image...")
    # Temp folder
    chdir(TEMP_ROOT)
    rmtree(path.split(TEMP_DIRECTORY)[0], ignore_errors=True)
    makedirs(path.join(TEMP_ROOT, TEMP_DIRECTORY))
    # Output folder
    logger.info("Generating output filepath for storage...")
    if not path.exists(ARCHIVE_DIRECTORY):
        makedirs(ARCHIVE_DIRECTORY)
    logger.info("Done.")

    # Take image
    logger.info("Taking image with camera...")
    with picamera.PiCamera() as camera:
        # Get her up and running
        camera.start_preview()
        logger.debug("Zzzz. Camera warm-up.")
        sleep(2)  # Warm up time 
        camera.resolution = (1024,768)
        camera.vflip = True

        filename = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        camera.capture(path.join(TEMP_ROOT, TEMP_DIRECTORY, filename+".jpg"), 'jpeg')
    logger.info("Done.")

    if classify:
        # Create VGG 
        logging.info("Generating VGG model... patience please.")
        vgg = vgg16.Vgg16()
        vgg.compile()
        logging.info("Your patience is rewarded. Done.")

        # Classify image
        logging.info("Classifying the image...")
        gen = ImageDataGenerator()
        test = gen.flow_from_directory(
            TEMP_DIRECTORY.split("/")[0],
            target_size=(224,224),
            batch_size=1,
            class_mode=None,
            shuffle=False,
        ) 
        predictions = vgg.model.predict_generator(generator=test, val_samples=1)
        logging.info("Done. Most likely outcome was #{}".format(predictions.argmax()))
        
        # Load the classes
        imagenet_classes_location = get_file("imagenet_class_index.json", "files.fast.ai/models/imagenet_class_index.json", cache_subdir='models')
        with open(imagenet_classes_location, 'r') as f:
            imagenet_classes = json.load(f)
        logging.info("Classname: {}".format(imagenet_classes[str(predictions.argmax())]))

        # Archive results and predictions
        logging.info("Archiving prediction results.")
        save(path.join(ARCHIVE_DIRECTORY, filename+".npy"), predictions)

        with open(path.join(ARCHIVE_DIRECTORY, filename+" results.csv"), 'a') as f:
            f.write("index, class, probability\n")
            for i in range(1000):
                f.write("{},{},{}\n".format(str(i), imagenet_classes[str(i)][1], predictions[0][i]))

    # Archive 
    logger.info("Archiving captured image...")
    copy(
        path.join(TEMP_ROOT, TEMP_DIRECTORY, filename+".jpg"),
        path.join(ARCHIVE_DIRECTORY, filename)
    )
    logger.info("Done.")


if __name__ == "__main__":

    # Parse arguments
    PROGRAM_DESCRIPTION = "Capture and classify an image."
    parser = argparse.ArgumentParser(PROGRAM_DESCRIPTION)
    parser.add_argument('-v', '--verbosity', action="count", help="Enable verbose output to file", default=0)
    parser.add_argument('-b', '--brainless', action="store_false", help="Run without neural net", default=True)
    args = parser.parse_args()

    # Establish logging
    logger = establish_logging(verbosity=args.verbosity)

    # Entry point goes here
    main(classify=args.brainless)
