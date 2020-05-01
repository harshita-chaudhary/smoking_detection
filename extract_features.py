"""
This script generates extracted features for each video, which other
models make use of.
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


# Set defaults.
seq_length = 50
class_limit = None  # Number of classes to extract. Can be 1-101 or None for all.

# Get the dataset.
data = DataSet(seq_length=seq_length, class_limit=class_limit)

# get the model.

# model = Extractor(weights="data/checkpoints/inception.035-0.17.hdf5")

# model = Extractor(weights="data/checkpoints/inception.009-0.29.hdf5")

model = Extractor(weights="data/checkpoints/inception.hdf5")

# Loop through data.
# print(data.data)
pbar = tqdm(total=len(data.data))
for video in data.data:

    # Get the path to the sequence for this video.
    path = os.path.join('data', 'sequences', video[2] + '-' + str(seq_length) + \
        '-features')  # numpy will auto-append .npy

    # Check if we already have it.
    if os.path.isfile(path + '.npy'):
        pbar.update(1)
        continue

    # Get the frames for this video.
    frames = data.get_frames_for_sample(video)

    # Now downsample to just the ones we need.
    frames = data.rescale_list(frames, seq_length)

    # Now loop through and extract features to build the sequence.
    sequence = []
    for image in frames:
        features = model.extract(image)
        sequence.append(features)

    dir = os.path.join('data', 'sequences')
    if not (os.path.exists(dir)):
        os.mkdir(dir)
    #Save the sequence.
    np.save(path, sequence)

    pbar.update(1)

pbar.close()
