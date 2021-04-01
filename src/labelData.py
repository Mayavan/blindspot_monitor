# %%
import cv2
import argparse
import os
from pathlib import Path

# %%
import json
with open("../label_data_config.json") as json_file:
            config = json.load(json_file)

file_path = config["file_path"]
data_dir = config["data_dir"]
prepare_frames = config["prepare_frames"]


try:
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


if(prepare_frames):
    vidcap = cv2.VideoCapture(file_path)
    success, image = vidcap.read()
    count = 0
    print("Preparing Frames")
    while success:
        cv2.imwrite("{}/{}.jpg".format(data_dir, count),
                    image)     # save frame as JPEG file
        success, image = vidcap.read()
        count += 1
# %%
count = 0
skip = 0
label = {}

while True:
    image = cv2.imread("{}/{}.jpg".format(data_dir, count))
    if(image is None):
        break

    cv2.imshow("Frame", image)
    print("Current frame number {}".format(count))
    key = cv2.waitKey(0)
    if key == ord('5'):
        # Mark 1 frame as false
        label["{}.jpg".format(count)] = 0
        count += 1
    elif key == ord('6'):
        # Mark 30 frames as false
        for i in range(30):
            label["{}.jpg".format(count)] = 0
            count += 1
    elif key == ord('2'):
        # Mark current frame as true
        label["{}.jpg".format(count)] = 1
        count += 1
    elif key == ord('1'):
        # Mark 30 frames as true
        for i in range(30):
            label["{}.jpg".format(count)] = 1
            count += 1
    elif key == ord('7'):
        # Go back 30 frames
        count -= 30
    elif key == ord('8'):
        # Go back 1 frames
        count -= 1
    elif key == ord('x'):
        # Exit
        break
    elif key == ord('d'):
        pass

# %%
print("Arranging Data")

percent_train = 60
percent_test = 20
percent_val = 20

count = 0

for fileName, cur_label in label.items():
    # Find sub directory
    if(count % 100 < percent_train):
        sub_dir = "train"
    elif(count % 100 < percent_test+percent_train):
        sub_dir = "test"
    else:
        sub_dir = "val"

    # Find class
    if(cur_label == 1):
        # Occupied class
        class_name = "occupied"
    else:
        # Free class
        class_name = "free"

    if not os.path.exists("{}/{}/{}/".format(data_dir, sub_dir, class_name)):
        os.makedirs("{}/{}/{}/".format(data_dir, sub_dir, class_name))

    if(Path("{}/{}".format(data_dir, fileName)).is_file()):
        Path("{}/{}".format(data_dir, fileName)
             ).rename("{}/{}/{}/{}".format(data_dir, sub_dir, class_name, fileName))
    else:
        print("{}/{}".format(data_dir, fileName) + " does not exist")

    count += 1



# %%
