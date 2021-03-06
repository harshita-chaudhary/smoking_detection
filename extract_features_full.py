"""
This script generates extracted features for each video, which other
models make use of.

You can change you sequence length and limit to a set number of classes
below.

class_limit is an integer that denotes the first N classes you want to
extract features from. This is useful is you don't want to wait to
extract all 101 classes. For instance, set class_limit = 8 to just
extract features for the first 8 (alphabetical) classes in the dataset.
Then set the same number when training models.
"""
import numpy as np
import os.path
from data_processor import DataSet
from extractor import Extractor
from tqdm import tqdm


# from keras.backend.tensorflow_backend import set_session
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
# sess = tf.Session(config=config)
# set_session(sess)
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

def extract_full_features(weights, seq_length = 40):
    # Set defaults.

    class_limit = None  # Number of classes to extract. Can be 1-101 or None for all.

    # Get the dataset.
    data = DataSet(seq_length=seq_length, class_limit=class_limit, check_dir='data/check')

    # get the model.
    # model = Extractor()
    # model = Extractor(weights="data/checkpoints/inception.009-0.29.hdf5")
    model = Extractor(weights)

    # Loop through data.
    print(data.data)
    pbar = tqdm(total=len(data.data))
    for video in data.data:

        # Get the path to the sequence for this video.
        path = os.path.join('data', 'sequences_test', video[2] + '-' + str(seq_length) + \
            '-features')  # numpy will auto-append .npy

        # Check if we already have it.
        if os.path.isfile(path + '.npy'):
            pbar.update(1)
            continue

        # Get the frames for this video.
        frames = data.get_frames_for_sample(video)

        # Now downsample to just the ones we need.
        # frames = data.rescale_list(frames, seq_length)

        # Now loop through and extract features to build the sequence.
        sequence = []
        for image in frames:
            features = model.extract(image)
            sequence.append(features)
        # print(path)
        output_dir = os.path.join('data', 'sequences_test')
        if not (os.path.exists(output_dir)):
            # create the directory you want to save to
            os.mkdir(output_dir)
        # Save the sequence.
        np.save(path, sequence)

        pbar.update(1)

    pbar.close()

