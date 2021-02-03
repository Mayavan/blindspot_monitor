from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers, losses
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Conv2D, AveragePooling2D, ZeroPadding2D
from tensorflow.keras.layers import Input, Add, Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import os
import argparse

weight_decay = 0.0005
save_dir = os.path.join(os.getcwd(), '../keras_models')

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Name of model to create.")
args = parser.parse_args()

class CreateNNGraph:
    def __init__(self, name):
        self.input_shape = (256, 256, 3)
        if name == "custom.h5":
            model = self.createModel()
        elif name == "AlexNet.h5":
            model = self.createModel2()
        elif name == "VGG16.h5":
            vgg16 = tf.keras.applications.VGG16(
                include_top=True,
                weights="imagenet",
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
            )
            model = Sequential()
            model.add(vgg16)
            model.add(Dense(2))
            model.add(Activation('softmax'))
        elif name == "inceptionv4.h5":
            inceptionv4 = tf.keras.applications.InceptionResNetV2(
                include_top=True,
                weights="imagenet",
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
            )
            model = Sequential()
            model.add(inceptionv4)
            model.add(Dense(2))
            model.add(Activation('sigmoid'))
        elif name == "xception.h5":
            xception = tf.keras.applications.Xception(
                include_top=True,
                weights="imagenet",
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
            )
            model = Sequential()
            model.add(xception)
            model.add(Dense(2))
            model.add(Activation('sigmoid'))
        else:
            print("Build Failed")
            return

        model.summary()
        print("Built: "+ name)

        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        model_path = os.path.join(save_dir, name)

        model.save(model_path)

    def createModel(self):
        model = Sequential()
        model.add(Conv2D(32, 3, input_shape=self.input_shape, strides=(1, 1), padding='valid', data_format=None,
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, 3, strides=(1, 1), padding='valid', data_format=None,
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

        model.add(Conv2D(64, 3, input_shape=self.input_shape, strides=(1, 1), padding='valid', data_format=None,
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, 3, strides=(1, 1), padding='valid', data_format=None,
                         dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                         bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(10))
        model.add(Activation('softmax'))

        return model


    def createModel2(self):

        # (3) Create a sequential model
        model = Sequential()

        # 1st Convolutional Layer
        model.add(Conv2D(filters=96, input_shape=self.input_shape, kernel_size=(10, 10),
                         strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(25, 25), strides=(4, 4), padding='valid'))
        # Batch Normalisation before passing it to the next layer
        model.add(BatchNormalization())

        # 2st Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(5, 5),
                         strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(15, 15), strides=(3, 3), padding='valid'))
        # Batch Normalisation before passing it to the next layer
        model.add(BatchNormalization())

        # 4th Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(10, 10), strides=(1, 1), padding='valid'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # Passing it to a dense layer
        model.add(Flatten())
        # 1st Dense Layer
        model.add(Dense(2048))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 2nd Dense Layer
        model.add(Dense(2048))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 3rd Dense Layer
        model.add(Dense(1000))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())

        # Output Layer
        model.add(Dense(2))
        model.add(Activation('softmax'))

        return model


if __name__ == "__main__":
    CreateNNGraph(args.model)

