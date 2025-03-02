import tensorflow as tf
import numpy as np
from image_classification_model import CNNClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

train_path='train_data/'
test_path='test_path/'

X = []
Y = []

def load_img(img):
    img=tf.io.read_file(img)
    img=tf.image.decode_jpeg(img,channels=1)
    img=tf.cast(img,tf.float32)/255
    img=tf.image.resize(img,(64,64))
    return img

def import_folder(path,max_count=100):
    x_data,y_data=[],[]
    sdasd=os.listdir(train_path)
    for animal_folder in os.listdir(train_path):
        count=0
        for image_path in os.listdir(os.path.join(train_path,animal_folder)):
            if count>max_count:
                break
            x_data.append(load_img(os.path.join(train_path,animal_folder,image_path)))
            y_data.append(str.lower(animal_folder))
            count+=1

    return np.array(x_data),np.array(y_data)



x_train,y_train=import_folder(train_path,200)
x_test,y_test=import_folder(test_path)
print('Importing is finished.')


labelencoder=LabelEncoder()
y_train=labelencoder.fit_transform(y_train)
y_test=labelencoder.transform(y_test)
print('LabelEncoding is finished.')


model=CNNClassifier()

print('\nStart training.')
model.train(x_train,y_train)
y_pred=model.predict(x_test)
model.save('image_classification_model.h5')
acc_score = accuracy_score(y_test, y_pred)
print(f'Accuracy score for test data: {acc_score}')

