import numpy as np
import datetime
import tensorflow as tf
import os
import signal
import json

model_dir = os.path.join(os.getcwd(), './keras_models')
logs_dir = os.path.join(os.getcwd(), './logs')


class TerminateOnFlag(tf.keras.callbacks.Callback):
    def __init__(self):
        self.terminate_flag = 0

    """Callback that terminates training when flag=1 is encountered."""

    def on_epoch_end(self, epoch, logs=None):
        print("\nEpoch has ended")
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

    def __init__(self, config):
        self.config = config
        signal.signal(signal.SIGINT, self.handler)
        model_path = os.path.join(model_dir, self.config["model"])
        self.initDataGenerator()
        self.model = tf.keras.models.load_model(model_path)

    def compileModel(self):
        adam = tf.keras.optimizers.Adam(
            learning_rate=self.config["optimizer"]["learning_rate"],
            beta_1=self.config["optimizer"]["beta_1"],
            beta_2=self.config["optimizer"]["beta_2"],
            epsilon=self.config["optimizer"]["epsilon"], amsgrad=False
        )
        self.model.compile(loss='binary_crossentropy', optimizer=adam,
                           metrics=['accuracy'])

    def initDataGenerator(self):
        # create a data generator
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=3,
                                                                  width_shift_range=10.,
                                                                  height_shift_range=10.,
                                                                  rescale=1.0/255.0,
                                                                  horizontal_flip=True)
        # load and iterate training dataset
        self.train_it = datagen.flow_from_directory(
            self.config["data"]+'/train/', class_mode='categorical', batch_size=64, target_size=(128, 128))
        # load and iterate validation dataset
        self.val_it = datagen.flow_from_directory(
            self.config["data"]+'/val/', class_mode='categorical', batch_size=64, target_size=(128, 128))
        # load and iterate test dataset
        self.test_it = datagen.flow_from_directory(
            self.config["data"]+'/test/', class_mode='categorical', batch_size=64, target_size=(128, 128))

    # Fit the model
    def trainModel(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.InteractiveSession(config=config)
        tf.compat.v1.keras.backend.set_session(session)

        log_dir = logs_dir + "/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)
        self.terminate_on_flag = TerminateOnFlag()

        history = self.model.fit(self.train_it,
                                 steps_per_epoch=20,
                                 validation_data=self.val_it,
                                 validation_steps=3,
                                 epochs=self.config["epochs"],
                                 callbacks=[tensorboard_callback,
                                            self.terminate_on_flag],
                                 verbose=1)

        self.__saveModel()
        tf.keras.backend.clear_session()

    # evaluate the model
    def testModel(self):
        loss, acc = self.model.evaluate(self.test_it, verbose=1)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

    def __saveModel(self):
        print("Saving Updated Model:")
        model_path = os.path.join(
            model_dir, self.config["model"][:-3] + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + ".h5")
        self.model.save(model_path)
        with open(model_path[:-2] + "json", "+w") as outfile:
            json.dump(self.config, outfile)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path of json config file")
    parser.add_argument("--data", help="Path of Video file to label")
    parser.add_argument("--model", help="Name of model in keras_models")
    args = parser.parse_args()

    config = {}

    # Default number of epochs
    config["epochs"] = 2

    # Default batch size
    config["batch_size"] = 100

    # Default data path
    config["data"] = '../data'

    # Default model name
    config["model"] = 'inceptionv4.h5'

    if args.config:
        with open(args.config) as json_file:
            config = json.load(json_file)

    # Override data path
    if args.data:
        config["data"] = args.data

    # Override model name
    if args.model:
        config["model"] = args.model

    net = NeuralNetwork(config)
    net.compileModel()
    net.trainModel()
