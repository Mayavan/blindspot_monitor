import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from matplotlib import pyplot
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

weight_decay = 0.0005
model_dir = os.path.join(os.getcwd(), '../keras_models')
logs_dir = os.path.join(os.getcwd(), '../logs')
results_dir = os.path.join(os.getcwd(), '../results/')


class NeuralNetwork:
    def __init__(self, name):
        self.name = name
        model_path = os.path.join(model_dir, name)
        self.model = load_model(model_path)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        # normalize inputs from 0-255 to 0.0-1.0
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

        # one hot encode outputs
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

    # Fit the model
    def trainModel(self, epochs, batch_size):
        generator = ImageDataGenerator(rotation_range=10,
                                       width_shift_range=5. / 32,
                                       height_shift_range=5. / 32,
                                       horizontal_flip=True)

        log_dir = logs_dir + "/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        generator.fit(self.x_train, seed=0, augment=True)
        history = self.model.fit(generator.flow(self.x_train, self.y_train),
                                           steps_per_epoch=len(self.x_train) // batch_size + 1,
                                           validation_data=(self.x_test, self.y_test),
                                           validation_steps=self.x_test.shape[0] // batch_size,
                                           epochs=epochs,
                                           callbacks=[tensorboard_callback],
                                           verbose=1)

        print("Saving Updated Model:")
        model_path = os.path.join(model_dir, self.name)
        self.model.save(model_path)

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




if __name__ == "__main__":
    # NeuralNetwork("ResNet56.h5").trainModel(100, 100)
    # NeuralNetwork("wrn.h5").trainModel(50, 100)
    # NeuralNetwork("LeNet.h5").trainModel(100, 500)
    # NeuralNetwork("custom.h5").trainModel(100, 500)
    net = NeuralNetwork("AlexNet.h5")
    net.trainModel(1, 100)

    print("model.inputs : ", net.model.inputs)
    print("model.outputs : ", net.model.outputs)
    # NeuralNetwork("wrn.h5").testModel()
