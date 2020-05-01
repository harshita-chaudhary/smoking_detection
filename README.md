## How to run the model to test the samples

### Testing one video using standalone script:

Dowwnload the script https://github.com/harshita-chaudhary/smoking_detection/blob/master/test.sh

Run it:  sh test.sh

### Testing one video:

python test.py --video_name [video_name]

Output:

•	timeLabel.json: JSON file containing the time in seconds vs probability of smoking action

•	timeLabel.jpg: Image showing action label. 0-non-smoking, 1-smoking

•	[video_name].avi: Video overlaid with label and probability of each frame


### Testing multiple videos:

Testing multiple videos:
1.	Place the videos to be tested in data/check folder under smoking and non-smoking directories. This is done to ensure that we remember which class the testing video belongs to. 

2.	Set the model weights in ‘validate_model.py’ file to point to the corresponding saved model in data/checkpoints directory or the downloaded model.

3.	Run ‘validate_model.py’ to classify the videos using the LSTM model. This generates a plot for each video with x-axis as the frame number and y-axis as the corresponding generated label. Since the videos used for testing are clipped videos and contain smoking or non-smoking action in entirety, the graph shows a straight line at either 0 or 1. 1 denotes smoking and 0 denotes non-smoking.

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
ActivityNet200 dataset link: http://activity-net.org/download.html
Kinetics700 dataset link: https://deepmind.com/research/open-source/kinetics

## Training models

TThe CNN model is trained by running `train_cnn.py`. This trains the InceptionV3 model with initial weights taken from the model trained on ImageNet. The model is further train on the HMDB data that is present in the data/train and data/test directories.

The RNN model is trained by running `train.py`. This trains the LSTM layers follwed by dense layers. 
The trained weights are stored in data/chekcpoints. This is used while classifying the videos.

The trained weights are stored in "data/chekcpoints". This is used while extracting the features.

## Demo/Using models

Demo on how to run is uploaded on Youtube.

Link: https://www.youtube.com/watch?v=VDf8s8x4WLA&list=PLFrrF91jLrRZhb-3Dcq8wIwYgWP-t0pFG&index=6

## References
http://jeffdonahue.com/lrcn/

https://keras.io/models/about-keras-models/

https://blog.coast.ai/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5

https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#dataset

http://activity-net.org/download.html

https://deepmind.com/research/open-source/kinetics

https://keras.io/models/about-keras-models/

https://sites.google.com/view/anxiaojiang/csce636
