## How to run the model to test the samples

The features have been extracted using CNN and saved in data/sequences directory in the .npy format.
These are corresponding to the five videos uploaded on Youtube - 3 smoking, 2 non-smoking

The following steps were followed to extract the features:
1. Move the test videos to data/test directory.
1. Run data/2_extract_file.py : This extracts frames from all the files in the data/train and data/test directories, stores them in the same directory and creates a csv file with the names of the files and number of frame present in the video.
Sample entries in the csv file:
test,non_smoking,ChikiMovie_ride_horse_f_cm_np1_fr_med_1,290
test,smoking,nice_smoking_girl_smoke_h_nm_np1_le_med_0,155
1. Extract features for each video using InceptionV3 by running extract_features_full.py. This stores the features in data/sequences directory.
1. Run validate_rnn_modified.py to classify the videos using the LSTM model. This generates a plot for each video with x-axis as the frame number and y-axis as the corresponding generated label. Since the videos used for testing are clipped videos and contain smoking or non-smoking action in entirity, the graph shows a straight line at either 0 or 1. 1 denotes smoking and 0 denotes non-smoking.

To run the model on the 5 sample videos provided in Youtube (https://www.youtube.com/playlist?list=PLFrrF91jLrRZhb-3Dcq8wIwYgWP-t0pFG), just run `python validate_rnn_modified.py`.

## Requirements

1.	Python 3
1.	Keras
1.	Tensorflow
1.	Numpy
1.	ffmpeg
1.	Matplotlib
1.	tqdm
1.	Pillow


## Getting the data

The data was taken from the HMDB51 dataset that contains clipped videos of 51 different actions.
For smoking, all the videos in the smoking class were taken, and for non-smoking, videos were randomly picked from the different action action classes to create an equally distributed dataset.

HMDB51 dataset link: https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#dataset

## Training models

The RNN model is trained by running `train.py`. This trains the LSTM layers follwed by dense layers. 
The trained weights are stored in data/chekcpoints. This is used while classifying the videos.

TThe CNN model is trained by running `train_cnn.py`. This trains the InceptionV3 model with initial weights taken from the model trained on ImageNet. The model is further train on the HMDB data that is present in the data/train and data/test directories.

The trained weights are stored in "data/chekcpoints". This is used while extracting the features.

## Demo/Using models

Demo on how to run is uploaded on Youtube.
Link: https://www.youtube.com/watch?v=VDf8s8x4WLA&list=PLFrrF91jLrRZhb-3Dcq8wIwYgWP-t0pFG&index=6

## References
http://jeffdonahue.com/lrcn/

https://keras.io/models/about-keras-models/

https://blog.coast.ai/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5



