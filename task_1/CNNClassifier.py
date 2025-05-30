from MnistClassifierInterface import MnistClassifierInterface
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Flatten,Input, Conv2D,MaxPooling2D
import numpy as np
class CNNClassifier(MnistClassifierInterface):
    """
    Convolution Neural Network model
    """

    def __init__(self):
        input = Input(shape=(8,8,1), dtype=tf.float64)

        conv1 = Conv2D(16,(3,3), activation='relu', name='conv_layer_1')(input)
        pool1 = MaxPooling2D((1,1),strides=None, padding="valid", name='maxpool_layer_1')(conv1)
        conv2 = Conv2D(32, (3, 3), activation='relu', name='conv_layer_2')(pool1)
        flatten=Flatten()(conv2)
        dense = Dense(256, activation='relu', name='dense_layer_2')(flatten)
        dense = Dense(128, activation='relu', name='dense_layer_5')(dense)
        dense = Dense(64, activation='relu', name='dense_layer_6')(dense)
        output = Dense(10, activation='softmax', name='output_layer')(dense)

        self.model = Model(inputs=input, outputs=output)

        self.model.summary()

    def train(self,x_train,y_train):
        x_train=x_train.reshape((x_train.shape[0],8, 8))
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x_train, y_train,epochs=30, batch_size=32)

    def predict(self,x_test):
        x_test = x_test.reshape((x_test.shape[0], 8, 8))
        self.model.predict(x_test)
        y_pred = self.model.predict(x_test)
        return np.argmax(y_pred, axis=1)

