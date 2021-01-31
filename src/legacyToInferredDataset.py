
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="Path to directory")
parser.add_argument("--out", help="Path to output directory")
args = parser.parse_args()


if not args.dir:
    print("Please input path to directory")
    exit()
else:
    location = args.dir


if not args.out:
    print("Please input path to directory")
    exit()
else:
    out_location = args.out

labels = '\n' + open(location + '/label.txt').read()

start = 0
while True:
    mid = labels.find(" ", start)
    endPosition = labels.find("\n", mid)
    try:
        fileName = labels[start+1 : mid]
        y = int(labels[mid: endPosition])
    except:
        continue
    start = endPosition

    if(Path(location + "/" + fileName).is_file()):
        if(y == 1):
            os.rename(location + "/" + fileName, out_location + "/occupied/"+ fileName)
        else:
            os.rename(location + "/" + fileName, out_location + "/free/"+ fileName)
    else:
        print(fileName + " does not exist")

