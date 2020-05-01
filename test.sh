#!/bin/bash
#Replace the variables with your github repo url, repo name, test
video name, json named by your UIN
GIT_REPO_URL="https://github.com/harshita-chaudhary/smoking_detection.git"
REPO="smoking_detection"
VIDEO="test_video1.mp4"
UIN_JSON="529005682.json"
UIN_JPG="529005682.jpg"
git clone $GIT_REPO_URL
cd $REPO
#Replace this line with commands for running your test python file.
echo $VIDEO
pip install -r requirements.txt
./download_models.sh
python test.py --video_name $VIDEO
#rename the generated timeLabel.json and figure with your UIN.
cp timeLable.json $UIN_JSON
cp timeLable.jpg $UIN_JPG
