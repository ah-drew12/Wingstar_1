import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Flatten,Input, Conv2D,MaxPooling2D,Conv3D,MaxPooling3D
import numpy as np
from keras.models import load_model
class CNNClassifier:
    """
    Convolution Neural Network model
    """

    def __init__(self,path=None):
        if path is None:
            input = Input(shape=(64,64,1), dtype=tf.float32)

            conv1 = Conv2D(32, (3, 3), activation='relu', name='conv_layer_1')(input)
            pool1 = MaxPooling2D((2, 2), name='maxpool_layer_1')(conv1)
            conv2 = Conv2D(64, (3, 3), activation='relu', name='conv_layer_2')(pool1)
            pool2 = MaxPooling2D((2, 2), name='maxpool_layer_2')(conv2)
            conv3 = Conv2D(128, (3, 3), activation='relu', name='conv_layer_3')(pool2)
            pool3 = MaxPooling2D((2, 2), name='maxpool_layer_3')(conv3)
            conv4 = Conv2D(256, (3, 3), activation='relu', name='conv_layer_4')(pool3)
            pool4 = MaxPooling2D((2, 2), name='maxpool_layer_4')(conv4)

            # Полносвязные слои
            flatten = Flatten()(pool2)
            dense1 = Dense(512, activation='relu', name='dense_layer_1')(flatten)
            dense2 = Dense(256, activation='relu', name='dense_layer_2')(dense1)
            dense3 = Dense(128, activation='relu', name='dense_layer_3')(dense2)
            output_layer = Dense(15, activation='softmax', name='output_layer')(dense3)
            self.model = Model(inputs=input, outputs=output_layer)
            self.model.summary()
        else:
            self.model=load_model('image_classification_model.h5')
    def train(self,x_train,y_train):
        x_train=x_train.reshape((x_train.shape[0],64, 64,1))
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x_train, y_train,epochs=10, batch_size=32)

    def predict(self,x_test):
        x_test = x_test.reshape((x_test.shape[0], 64, 64,1))
        y_pred = self.model.predict(x_test)
        return np.argmax(y_pred, axis=1)

    def save(self,path):
        self.model.save(path)


