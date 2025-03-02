from MnistClassifier import MnistClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf

data = load_digits()
X = data.data/255
Y = data.target

x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.3)

model=MnistClassifier(algorithm='rf')
model.train(x_train,y_train)
y_pred=model.predict(x_test)
acc_score = accuracy_score(y_test, y_pred)
print(f'Accuracy score for test data: {acc_score}')
print('a')