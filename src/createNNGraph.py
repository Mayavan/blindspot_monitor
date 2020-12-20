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
import os

weight_decay = 0.0005
save_dir = os.path.join(os.getcwd(), '../keras_models')


class CreateNNGraph:
    def __init__(self, name):
        self.input_shape = (32, 32, 3)
        if name == "LeNet.h5":
            model = self.createLeNet()
            print("Built: LeNet")
        elif name == "ResNet56.h5":
            self.input_shape = (32, 32, 3)
            model = self.resnet_v2(self.input_shape, depth=56, num_classes=10)
            model.summary()
            print("Built: ResNet56")
        elif name == "VGG19.h5":
            model = vgg16.VGG16(include_top=False, weights=None, input_tensor=None, input_shape=self.input_shape, pooling=None,
                        classes=1000)
            model.summary()
            print("Built: VGG19")
        elif name == "wrn.h5":
            init = (32, 32, 3)
            model = self.create_wide_residual_network(init, nb_classes=10, N=4, k=8, dropout=0.0)
            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
            model.summary()
            print("Built: WRN")
        elif name == "custom.h5":
            model = self.createModel()
            print("Built: Custom")
        elif name == "AlexNet.h5":
            model = self.createAlexNet()
            print("Built: AlexNet")
        else:
            print("Build Failed")
            return


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

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        model.summary()
        return model

    def createLeNet(self):
        model = Sequential()
        model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=self.input_shape, padding="same"))
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
        model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        model.add(Flatten())
        model.add(Dense(84, activation='tanh'))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=losses.categorical_crossentropy, optimizer='SGD', metrics=['acc'])
        model.summary()
        return model

    def resnet_v2(self, input_shape, depth, num_classes=10):
        """ResNet Version 2 Model builder [b]

        Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
        bottleneck layer
        First shortcut connection per layer is 1 x 1 Conv2D.
        Second and onwards shortcut connection is identity.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filter maps is
        doubled. Within each stage, the layers have the same number filters and the
        same filter map sizes.
        Features maps sizes:
        conv1  : 32x32,  16
        stage 0: 32x32,  64
        stage 1: 16x16, 128
        stage 2:  8x8,  256

        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)

        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 9 != 0:
            raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
        # Start model definition.
        num_filters_in = 16
        num_res_blocks = int((depth - 2) / 9)

        inputs = Input(shape=input_shape)
        # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
        x = self.resnet_layer(inputs=inputs,
                         num_filters=num_filters_in,
                         conv_first=True)

        # Instantiate the stack of residual units
        for stage in range(3):
            for res_block in range(num_res_blocks):
                activation = 'relu'
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:  # first layer and first stage
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:  # first layer but not first stage
                        strides = 2  # downsample

                # bottleneck residual unit
                y = self.resnet_layer(inputs=x,
                                 num_filters=num_filters_in,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=activation,
                                 batch_normalization=batch_normalization,
                                 conv_first=False)
                y = self.resnet_layer(inputs=y,
                                 num_filters=num_filters_in,
                                 conv_first=False)
                y = self.resnet_layer(inputs=y,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 conv_first=False)
                if res_block == 0:
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = self.resnet_layer(inputs=x,
                                     num_filters=num_filters_out,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False)
                x = keras.layers.add([x, y])

            num_filters_in = num_filters_out

        # Add classifier on top.
        # v2 has BN-ReLU before Pooling
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['acc'])
        return model

    def resnet_layer(self, inputs,
                     num_filters=16,
                     kernel_size=3,
                     strides=1,
                     activation='relu',
                     batch_normalization=True,
                     conv_first=True):
        """2D Convolution-Batch Normalization-Activation stack builder

        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            num_filters (int): Conv2D number of filters
            kernel_size (int): Conv2D square kernel dimensions
            strides (int): Conv2D square stride dimensions
            activation (string): activation name
            batch_normalization (bool): whether to include batch normalization
            conv_first (bool): conv-bn-activation (True) or
                bn-activation-conv (False)

        # Returns
            x (tensor): tensor as input to the next layer
        """
        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x

    weight_decay = 0.0005

    def initial_conv(self, input):
        x = Convolution2D(16, (3, 3), padding='same', kernel_initializer='he_normal',
                          W_regularizer=l2(weight_decay),
                          use_bias=False)(input)

        channel_axis = 1 if K.image_data_format() == "channels_first" else -1

        x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)
        return x

    def expand_conv(self, init, base, k, strides=(1, 1)):
        x = Convolution2D(base * k, (3, 3), padding='same', strides=strides, kernel_initializer='he_normal',
                          W_regularizer=l2(weight_decay),
                          use_bias=False)(init)

        channel_axis = 1 if K.image_data_format() == "channels_first" else -1

        x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)

        x = Convolution2D(base * k, (3, 3), padding='same', kernel_initializer='he_normal',
                          W_regularizer=l2(weight_decay),
                          use_bias=False)(x)

        skip = Convolution2D(base * k, (1, 1), padding='same', strides=strides, kernel_initializer='he_normal',
                             W_regularizer=l2(weight_decay),
                             use_bias=False)(init)

        m = Add()([x, skip])

        return m

    def conv1_block(self, input, k=1, dropout=0.0):
        init = input

        channel_axis = 1 if K.image_data_format() == "channels_first" else -1

        x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
        x = Activation('relu')(x)
        x = Convolution2D(16 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                          W_regularizer=l2(weight_decay),
                          use_bias=False)(x)

        if dropout > 0.0: x = Dropout(dropout)(x)

        x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)
        x = Convolution2D(16 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                          W_regularizer=l2(weight_decay),
                          use_bias=False)(x)

        m = Add()([init, x])
        return m

    def conv2_block(self, input, k=1, dropout=0.0):
        init = input

        channel_axis = 1 if K.image_dim_ordering() == "th" else -1

        x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
        x = Activation('relu')(x)
        x = Convolution2D(32 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                          W_regularizer=l2(weight_decay),
                          use_bias=False)(x)

        if dropout > 0.0: x = Dropout(dropout)(x)

        x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)
        x = Convolution2D(32 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                          W_regularizer=l2(weight_decay),
                          use_bias=False)(x)

        m = Add()([init, x])
        return m

    def conv3_block(self, input, k=1, dropout=0.0):
        init = input

        channel_axis = 1 if K.image_dim_ordering() == "th" else -1

        x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
        x = Activation('relu')(x)
        x = Convolution2D(64 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                          W_regularizer=l2(weight_decay),
                          use_bias=False)(x)

        if dropout > 0.0: x = Dropout(dropout)(x)

        x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)
        x = Convolution2D(64 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                          W_regularizer=l2(weight_decay),
                          use_bias=False)(x)

        m = Add()([init, x])
        return m

    def create_wide_residual_network(self, input_dim, nb_classes=100, N=2, k=1, dropout=0.0, verbose=1):
        """
        Creates a Wide Residual Network with specified parameters
        :param input: Input Keras object
        :param nb_classes: Number of output classes
        :param N: Depth of the network. Compute N = (n - 4) / 6.
                  Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
                  Example2: For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
                  Example3: For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
        :param k: Width of the network.
        :param dropout: Adds dropout if value is greater than 0.0
        :param verbose: Debug info to describe created WRN
        :return:
        """
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1

        ip = Input(shape=input_dim)

        x = self.initial_conv(ip)
        nb_conv = 4

        x = self.expand_conv(x, 16, k)
        nb_conv += 2

        for i in range(N - 1):
            x = self.conv1_block(x, k, dropout)
            nb_conv += 2

        x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)

        x = self.expand_conv(x, 32, k, strides=(2, 2))
        nb_conv += 2

        for i in range(N - 1):
            x = self.conv2_block(x, k, dropout)
            nb_conv += 2

        x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)

        x = self.expand_conv(x, 64, k, strides=(2, 2))
        nb_conv += 2

        for i in range(N - 1):
            x = self.conv3_block(x, k, dropout)
            nb_conv += 2

        x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)

        x = AveragePooling2D((8, 8))(x)
        x = Flatten()(x)

        x = Dense(nb_classes, W_regularizer=l2(weight_decay), activation='softmax')(x)

        model = Model(ip, x)

        if verbose: print("Wide Residual Network-%d-%d created." % (nb_conv, k))
        return model

    def createAlexNet(self):

        # (3) Create a sequential model
        model = Sequential()

        # 1st Convolutional Layer
        model.add(Conv2D(filters=96, input_shape=(32, 32, 3), kernel_size=(3, 3),
                         strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        # Batch Normalisation before passing it to the next layer
        model.add(BatchNormalization())

        # 2st Convolutional Layer
        model.add(Conv2D(filters=256, input_shape=(32, 32, 3), kernel_size=(3, 3),
                         strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        # Batch Normalisation before passing it to the next layer
        model.add(BatchNormalization())

        # 4th Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
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
        model.add(Dense(10))
        model.add(Activation('softmax'))

        model.summary()

        # (4) Compile
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
        return model


if __name__ == "__main__":
    # CreateNNGraph("wrn.h5")
    # CreateNNGraph("ResNet56.h5")
    # CreateNNGraph("LeNet.h5")
    # CreateNNGraph("custom.h5")
    CreateNNGraph("AlexNet.h5")

