"""
After moving all the files using the 1_ file, we run this one to extract
the images from the videos and also create a data file we can use
for training and testing later.
"""
import csv
import glob
import os
import os.path
from subprocess import call

def extract_files(folders=None):
    """After we have all of our videos split between train and test, and
    all nested within folders representing their classes, we need to
    make a data file that we can reference when training our RNN(s).
    This will let us keep track of image sequences and other parts
    of the training process.

    We'll first need to extract images from each of the videos. We'll
    need to record the following data in the file:

    [train|test], class, filename, nb frames

    Extracting can be done with ffmpeg:
    `ffmpeg -i video.mpg image-%04d.jpg`
    """
    data_file = []
    if folders is None:
        folders = ['train', 'test']

    for folder in folders:
        class_folders = glob.glob(os.path.join('data',folder, '*'))

        for vid_class in class_folders:
            class_files = glob.glob(os.path.join(vid_class, '*.avi'))
            class_files.extend(glob.glob(os.path.join(vid_class, '*.mp4')))
            print(class_files)
            for video_path in class_files:
                # Get the parts of the file.
                video_parts = get_video_parts(video_path)

                train_or_test, classname, filename_no_ext, filename, filename_ext = video_parts

                # Only extract if we haven't done it yet. Otherwise, just get
                # the info.
                if not check_already_extracted(video_parts):
                    # Now extract it.
                    src = os.path.join('data',train_or_test, classname, filename)
                    dest = os.path.join('data',train_or_test, classname,
                        filename_no_ext + '-%04d.jpg')
                    dirname = os.path.dirname(__file__)
                    # src = src.replace("\\", "/")
                    print(src)
                    filename = os.path.join(dirname, src)
                    # print(filename.replace("\\", "/")
                    print(filename)
                    print(dest)

                    if os.path.isfile(filename):
                        print("File exist")
                    if os.path.isfile(dest):
                        print("File dest exist")
                    # call(["ffmpeg", "-i", "r\"" + src + "\"", "r\"" + dest + "\""], shell=True)
                    call(["ffmpeg", "-i", src, dest], shell=True)
                # Now get how many frames it is.
                nb_frames = get_nb_frames_for_video(video_parts)

                data_file.append([train_or_test, classname, filename_no_ext, nb_frames, filename_ext])

                print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

    write_file = 'data_file.csv' if folders is None else 'data/data_test_file.csv'

    with open(write_file, 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)

    print("Extracted and wrote %d video files." % (len(data_file)))

def get_nb_frames_for_video(video_parts):
    """Given video parts of an (assumed) already extracted video, return
    the number of frames that were extracted."""
    train_or_test, classname, filename_no_ext, _, _ = video_parts
    generated_files = glob.glob(os.path.join('data',train_or_test, classname,
                                filename_no_ext + '*.jpg'))
    return len(generated_files)

def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    parts = video_path.split(os.path.sep)
    filename = parts[3]
    filename_no_ext = filename.split('.')[0]
    classname = parts[2]
    train_or_test = parts[1]
    filename_ext = filename.split('.')[1]
    return train_or_test, classname, filename_no_ext, filename, filename_ext

def check_already_extracted(video_parts):
    """Check to see if we created the -0001 frame of this file."""
    train_or_test, classname, filename_no_ext, _, _ = video_parts
    return bool(os.path.exists(os.path.join(train_or_test, classname,
                               filename_no_ext + '-0001.jpg')))

def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. It can have format:

    [train|test], class, filename, nb frames
    """
    extract_files()

if __name__ == '__main__':
    main()
