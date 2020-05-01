import argparse
import shutil
from subprocess import call
import cv2
from tqdm import tqdm
from data_processor import DataSet as data
from extractor import Extractor
import numpy as np
import glob
import os.path
import os
import sys
from keras.models import load_model
from matplotlib import pyplot as plt
import json
import math
import tensorflow as tf
sys.path.insert(1, 'data')

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

def main(args):
    video_name = args.video_name
    seq_length = 50
    # weights_path = get_file('lstm-features.hdf5', 'https://1drv.ms/u/s!AjwTYpyMoMlUll4oFEpdBU9dlppN?e=WVXhgu')
    # url = 'https://1drv.ms/u/s!AjwTYpyMoMlUll4oFEpdBU9dlppN?e=WVXhgu'
    # urllib.request.urlretrieve(url, 'data/checkpoints/lstm-features.hdf5')
    # model = load_model('data/checkpoints/lstm-features.009-0.454.hdf5')
    # model = load_model('data/checkpoints/lstm-features.004-0.614.hdf5')
    # model = load_model('data/checkpoints/lstm-features.017-0.849.hdf5')

    cnn_model = 'cnn.hdf5'
    lstm_model = 'lstm.hdf5'

    model = load_model(lstm_model)
    extractor = Extractor(weights=cnn_model)

    filename_no_ext = video_name.split('.')[0]
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    dest = os.path.join('tmp', filename_no_ext + '-%04d.jpg')
    call(["ffmpeg", "-i", video_name, "-filter:v", "fps=fps=30", dest], shell=True)
    generated_frames = glob.glob(os.path.join('tmp', filename_no_ext + '*.jpg'))
    nb_frames = len(generated_frames)
    print("Generated %d frames for %s" % (nb_frames, filename_no_ext))
    pbar = tqdm(total=len(generated_frames))

    # Now loop through and extract features to build the sequence.
    sequence = []
    for image in generated_frames:
        features = extractor.extract(image)
        sequence.append(features)
        pbar.update(1)
    pbar.close()
    sequence = np.asarray(sequence)

    shutil.rmtree('tmp')
    classes = []
    classes.append('smoking')
    classes.append('non_smoking')
    classes = sorted(classes)

    output_json = {"smoking": []}

    total = sequence.shape[0]
    frames = np.arange(total)
    frame_pred = np.ones(total)
    frame_pred_sum = np.zeros(total)
    frame_pred_count = np.zeros(total)
    # print("Size : " + str(total))
    frame_pred_prob = np.empty(total)
    frame_pred_prob[:] = np.nan
    skip_clips = 15  #100
    skip_frames = 2  #6
    start_frame = 0
    end_frame = skip_frames*seq_length
    # end_frame = 250
    print("Number of frames: ", total )
    label_predictions = {}
    if end_frame > sequence.shape[0]:
        sequences = data.rescale_list(sequence, seq_length)
        X = []
        X.append(sequences)
        predictions = model.predict(np.array(X), batch_size=1)
        label_predictions = {}

        for i, label in enumerate(classes):
            label_predictions[label] = predictions[0][i]
        # print(label_predictions)
        # if label_predictions["smoking"] <= 0.5:
            # frame_pred[start_frame:total] = 0
        for i in range(start_frame, total):
            frame_pred_sum[i] += label_predictions["smoking"]
            frame_pred_count[i] += 1
        # else:
        #     frame_pred[start_frame:total] = -1

        # frame_pred_prob[start_frame:total] = str(label_predictions["smoking"])

    else:
        while end_frame <= sequence.shape[0]:

            X = []
            x = []
            for i in range(start_frame, end_frame, skip_frames):
                x.append(sequence[i,:])
            X.append(x)
            # print("video: " + video[2] + " start frame: " + str(start_frame) + " end frame: " + str(end_frame))
            # X.append(sequences[start_frame: end_frame,:])
            # sequence = sequence.reshape(1, 3, 3)
            predictions = model.predict(np.array(X), batch_size=1)
            label_predictions = {}
            for i, label in enumerate(classes):
                # print(predictions)
                label_predictions[label] = predictions[0][i]
            # print(label_predictions)
            # if label_predictions["smoking"] <= 0.5:
            #     frame_pred[start_frame:end_frame] = 0
            for i in range(start_frame, end_frame):
                frame_pred_sum[i] += label_predictions["smoking"]
                frame_pred_count[i] += 1
            # else:
            #     frame_pred[start_frame:end_frame] = 0

            # frame_pred_prob[start_frame:end_frame] = str(label_predictions["smoking"])

            start_frame += skip_clips
            end_frame += skip_clips

        for i in range(start_frame, min(sequence.shape[0], end_frame-1)):
            frame_pred_sum[i] += label_predictions["smoking"]
            frame_pred_count[i] += 1
        #
        # for i in range(start_frame, min(sequences.shape[0], end_frame-1)):
        #     # frame_pred_prob.append(str(label_predictions["smoking"]))
        #     frame_pred_prob[i] = str(label_predictions["smoking"])
        #     if label_predictions["smoking"] <= 0.5:
        #         frame_pred[i] = 0
    # print(frame_pred)
    for i in range(0,total):
        frame_pred_prob[i] = frame_pred_sum[i]/frame_pred_count[i]
        if frame_pred_prob[i] < 0.5:
            frame_pred[i] = 0

    plt.title("Smoking action detection")
    plt.xlabel("Time (in seconds)")
    plt.ylabel("Smoking label")
    plt.plot(frames, frame_pred)
    plt.yticks(np.arange(2))
    xlabel_count = 10
    ticker_len = 1
    if len(frames)/30 > xlabel_count:
        ticker_len = math.ceil(len(frames)/(30*xlabel_count))
    plt.xticks(np.arange(min(frames), max(frames) + ticker_len, (30 * ticker_len)),
               np.arange(min(frames) / 30, max(frames) / 30 + 30, ticker_len))

    output_path = "timeLabel.jpg"
    print("Saving output labels to: ", output_path)

    plt.savefig(output_path)
    plt.close()
    # plt.show()
    # plt.figure()
    frame_time = [frame/30 for frame in frames]
    output_json["smoking"] = list(zip(frame_time, frame_pred_prob))
    y = json.dumps(output_json)
    # with open('frameLabel.json', 'w') as outfile:
    #     json.dump(y, outfile)
    output_path = "timeLabel.json"
    print(y)
    with open(output_path, 'w') as outfile:
        json.dump(output_json, outfile)
        print('Output JSON saved under {}'.format(output_path))
    label_video(video_name, frame_pred, frame_pred_prob)

def label_video(video_name, labels, label_prob):
    filename_no_ext = video_name.split('.')[0]
    print('Starting: {}'.format(filename_no_ext))
    start_frame = 0
    end_frame = None
    reader = cv2.VideoCapture(video_name)
    video_fn = filename_no_ext+'_output.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total number of frames ", str(num_frames))
    writer = None
    # Text variables
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 0.35

    # Frame numbers and length of output video
    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame-start_frame)

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1

        if frame_num < start_frame or frame_num-1 >= len(labels):
            continue
        pbar.update(1)

        # Image size
        height, width = image.shape[:2]

        # Init output writer
        if writer is None:
            writer = cv2.VideoWriter( video_fn, fourcc, fps,
                                     (height, width)[::-1])
        # print("frame num " + str(frame_num))
        label = 'smoking' if labels[frame_num-1] == 1 else 'non_smoking'
        color = (0, 255, 0) if labels[frame_num-1] == 0 else (0, 0, 255)
        # cv2.putText(image,  'Label =>' + str(label), (x, y + h + 30),
        #             font_face, font_scale,
        #             color, thickness, 2)
        cv2.putText(image, 'Frame: ' + str(frame_num) + ', Label: ' + str(label) +
                    ', Prob =>' + str(label_prob[frame_num-1]), (10, 20),
                    font_face, font_scale,
                    color, thickness, 2)
        # cv2.putText(image, 'Frame num: ' + str(frame_num) + ', Label =>' + str(label) +
        #             ', Prob =>' + label_prob[frame_num-1], (10, 20),
        #             font_face, font_scale,
        #             color, thickness, 2)
        if frame_num >= end_frame:
            break
        cv2.imshow('test', image)
        cv2.waitKey(1)     # About 30 fps
        writer.write(image)
    pbar.close()
    if writer is not None:
        writer.release()
        print('Finished! Output saved under {}'.format(video_fn))
    else:
        print('Input video file was empty')

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the smoking detection model.")
    parser.add_argument("-v", "--video_name", required=True, help="Path to testing video")
    parsed = parser.parse_args()
    main(parsed)

