import cv2
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--file", help="Path of Video file to label")
parser.add_argument("--out", help="Path of data directory")
parser.add_argument("--frames_from_folder", help="Use to use frames already in out directory", action="store_true")
args = parser.parse_args()

if(not args.frames_from_folder):
    if not args.file:
        file_path = '../../day_right_1.mp4'
    else:
        file_path = args.file


if not args.out:
    data_dir = '../data'
else:
    data_dir = args.out

if args.frames_from_folder:
    prepare_frames = False
else:
    prepare_frames = True


try:
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


if(prepare_frames):
    vidcap = cv2.VideoCapture(file_path)
    success,image = vidcap.read()
    count = 0
    print("Preparing Frames")
    while success:
        cv2.imwrite("{}/{}.jpg".format(data_dir,count), image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1

count = 0
skip = 0
label = {}

while True:
    image  = cv2.imread("{}/{}.jpg".format(data_dir,count))
    if(image is None):
        break

    cv2.imshow("Frame", image)
    print("Current frame number {}".format(count))
    key = cv2.waitKey(0)
    if key == ord('5'):
        # Mark 1 frame as false
        count += 1
    elif key == ord('6'):
        # Mark 30 frames as false
        for i in range(30):
            label["{}.jpg".format(count)] = 0
            count += 1
    elif key == ord('0'):
        # Mark current frame as true
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

f = open(data_dir+"/label.txt", "w")
for key, val in label.items():
    f.write("{} {}\n".format(key, val))

f.close()