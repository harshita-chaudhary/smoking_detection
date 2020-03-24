"""
Validate our RNN. Basically just runs a validation generator on
about the same number of videos as we have in our test set.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from models import ResearchModels
import numpy as np
import operator
import random
import glob
import os.path
from data import DataSet
from processor import process_image
from keras.models import load_model
from matplotlib import pyplot as plt
import json


def validate(data_type, model, seq_length=40, saved_model=None,
             class_limit=None, image_shape=None):
    batch_size = 32

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    val_generator = data.frame_generator(batch_size, 'test', data_type)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    # Evaluate!
    results = rm.model.evaluate_generator(
        generator=val_generator,
        val_samples=3200)

    print(results)
    print(rm.model.metrics_names)


def main(videos=5):

    model = load_model('data/checkpoints/lstm-features.020-0.366.hdf5')
    if model == 'conv_3d' or model == 'lrcn':
        data_type = 'images'
        image_shape = (80, 80, 3)
    else:
        data_type = 'features'
        image_shape = None

    # validate(data_type, model, saved_model=None,
    #          image_shape=image_shape, class_limit=4)

    output_json = {}
    output_json["smoking"] = []
    data = DataSet()
    # model = load_model('data/checkpoints/inception.057-1.16.hdf5')
    for video in data.data:
        X, y = [], []

        sequences = data.get_extracted_sequence(data_type, video)
        if sequences is None:
            print("Sequence data does not exist for ", video)
        total = sequences.shape[0]
        frames = np.arange(total)
        frame_pred = np.zeros(total)
        frame_pred_prob = []
        # X.append(sequence)
        y.append(data.get_class_one_hot(video[1]))

        end_frame = 40
        start_frame = 0
        # print("Number of frames: ", total )
        print("video: " + video[2])
        while end_frame < sequences.shape[0]:
            X = []
            X.append(sequences[start_frame: end_frame,:])
            # sequence = sequence.reshape(1, 3, 3)
            predictions = model.predict(np.array(X), batch_size=1)

            # Show how much we think it's each one.
            label_predictions = {}
            for i, label in enumerate(data.classes):
                # print(predictions)
                label_predictions[label] = predictions[0][i]
            # print(label_predictions)
            frame_pred_prob.append(str(label_predictions["smoking"]))
            if label_predictions["smoking"] > 0.5:
                frame_pred[start_frame:end_frame] = 1
            #
            # sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)
            # for i, class_prediction in enumerate(sorted_lps):
            #     # Just get the top five.
            #     # if i > 4:
            #     #     break
            #     print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
            #     i += 1
            start_frame += 1
            end_frame += 1
        # print(frame_pred)
        plt.title("Smoking action detection")
        plt.xlabel("Frame")
        plt.ylabel("Smoking action present")
        plt.plot(frames, frame_pred)
        plt.show()
        plt.figure()
        output_json["smoking"] = dict(zip(frames.tolist(), frame_pred_prob))
        y = json.dumps(output_json)
        # with open('frameLabel.json', 'w') as outfile:
        #     json.dump(y, outfile)

        print(y)

if __name__ == '__main__':
    main()
