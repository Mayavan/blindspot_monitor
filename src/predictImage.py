import os
import sys
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

weight_decay = 0.0005
save_dir = os.path.join(os.getcwd(), '../keras_models')

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Name of model to create.")
parser.add_argument("--image", help="Name of image to predict.")
args = parser.parse_args()

model_dir = os.path.join(os.getcwd(), '../keras_models')
if args.model:
    model_path = os.path.join(model_dir, args.model)
else:
    model_path = os.path.join(model_dir, "inceptionv4.h5")

model = tf.keras.models.load_model(model_path)

if args.image:
    img_path = args.image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    prediction = model.predict(img_preprocessed)
    print(img_path, end='')
    print(prediction)
    sys.exit(0)

for i in range(64):
    img_path = "dump/train/batch0/{}.png".format(i)
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    prediction = model.predict(img_preprocessed)
    print(img_path, end='')
    print(prediction)
