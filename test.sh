#!/bin/bash
#Replace the variables with your github repo url, repo name, test video name, json named by your UIN
GIT_REPO_URL="https://github.com/harshita-chaudhary/smoking_detection.git"
REPO="smoking_detection"
VIDEO="test_video1.mp4"
UIN_JSON="529005682.json"
UIN_JPG="529005682.jpg"
git clone $GIT_REPO_URL
cd $REPO
# cp ../$VIDEO .
#Replace this line with commands for running your test python file.
pip install -r requirements.txt
youtube-dl -f best -f mp4 "https://www.youtube.com/watch?v=OCT3Y3BhrLo" -o $VIDEO
# Install ffmpeg
sudo apt install ffmpeg
echo $VIDEO
chmod +x download_models.sh
./download_models.sh
python test.py --video_name $VIDEO
#rename the generated timeLabel.json and figure with your UIN.
cp timeLabel.json ../$UIN_JSON
cp timeLabel.jpg ../$UIN_JPG
