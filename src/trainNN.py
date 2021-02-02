import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import Callback
from matplotlib import pyplot
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import tensorflow as tf
import argparse
import os
import signal

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="Path of Video file to label")
parser.add_argument("--model", help="Name of model in keras_models")
args = parser.parse_args()

model_dir = os.path.join(os.getcwd(), '../keras_models')
logs_dir = os.path.join(os.getcwd(), '../logs')
results_dir = os.path.join(os.getcwd(), '../results/')

if not args.data:
    data_dir = '../data'
else:
    data_dir = args.data

if not args.model:
    args.model = 'inceptionv4.h5'


class TerminateOnFlag(Callback):
    def __init__(self, name):
        self.name = name
        self.terminate_flag = 0

    """Callback that terminates training when flag=1 is encountered."""
    def on_epoch_end(self, epoch, logs=None):
        print("\nEpoch has ended")
        print("\nTerminate flag = " + str(self.terminate_flag))
        if self.terminate_flag:
            print('\nStop training and save the model')
            self.model.stop_training = True

    def update_flag(self, val):
        self.terminate_flag = val

class NeuralNetwork:
    # Monitors the SIGINT (ctrl + C) to safely stop training when it is sent
    def handler(self, signum, frame):
        print('\nSignal handler called with signal', signum)
        self.terminate_on_flag.update_flag(1)


    def __init__(self, name):
        signal.signal(signal.SIGINT, self.handler)
        self.name = name
        model_path = os.path.join(model_dir, name)
        self.initDataGenerator()
        self.model = load_model(model_path)

    def initDataGenerator(self):
        # create a data generator
        datagen = ImageDataGenerator(rotation_range=5,
                                       width_shift_range=50.,
                                       height_shift_range=50.,
                                       rescale=1.0/255.0,
                                       horizontal_flip=False)
        # load and iterate training dataset
        self.train_it = datagen.flow_from_directory(data_dir+'/train/', class_mode='categorical', batch_size=64, target_size=(224, 224))
        # load and iterate validation dataset
        self.val_it = datagen.flow_from_directory(data_dir+'/val/', class_mode='categorical', batch_size=32, target_size=(224, 224))
        # load and iterate test dataset
        self.test_it = datagen.flow_from_directory(data_dir+'/test/', class_mode='categorical', batch_size=32, target_size=(224, 224))

    # Fit the model
    def trainModel(self, epochs, batch_size):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.InteractiveSession(config=config)
        tf.compat.v1.keras.backend.set_session(session)

        log_dir = logs_dir + "/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        self.terminate_on_flag = TerminateOnFlag(self.name)

        history = self.model.fit(self.train_it,
                                steps_per_epoch=20,
                                validation_data=self.val_it,
                                validation_steps=2,
                                epochs=epochs,
                                callbacks=[tensorboard_callback, self.terminate_on_flag],
                                verbose=1)

        self.__saveModel()

        print("Saving training graphs:")
        try:
            if not os.path.exists(results_dir+self.name[0: -3]):
                os.makedirs(results_dir+self.name[0: -3])
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        # plot metrics
        pyplot.title(''.join([self.name[0: -3], " with mini batch size of ", str(batch_size)]))
        pyplot.xlabel('Epochs')
        pyplot.ylabel('Loss')

        pyplot.plot(history.history['val_loss'], label='Validation Loss')
        pyplot.plot(history.history['loss'], label='Training Loss')
        pyplot.legend()
        pyplot.savefig(results_dir+self.name[0: -3]+'/loss.png')
        pyplot.show()

        pyplot.title(''.join([self.name[0: -3], " with mini batch size of ", str(batch_size)]))
        pyplot.xlabel('Epochs')
        pyplot.ylabel('Accuracy')
        pyplot.plot(history.history['val_accuracy'], label='Validation Accuracy')
        pyplot.plot(history.history['accuracy'], label='Training Accuracy')
        pyplot.legend()
        pyplot.savefig(results_dir+self.name[0: -3]+'/Accuracy.png')
        pyplot.show()

    # evaluate the model
    def testModel(self):
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

    def __saveModel(self):
        print("Saving Updated Model:")
        model_path = os.path.join(model_dir, self.name)
        self.model.save(model_path)




if __name__ == "__main__":
    # NeuralNetwork("ResNet56.h5").trainModel(100, 100)
    # NeuralNetwork("wrn.h5").trainModel(50, 100)
    # NeuralNetwork("LeNet.h5").trainModel(100, 500)
    # NeuralNetwork("custom.h5").trainModel(100, 500)
    #net = NeuralNetwork("AlexNet.h5")
    net = NeuralNetwork(args.model)
    net.trainModel(2, 100)

    print("model.inputs : ", net.model.inputs)
    print("model.outputs : ", net.model.outputs)
    # NeuralNetwork("wrn.h5").testModel()
