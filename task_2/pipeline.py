from sklearn.pipeline import Pipeline
from ner_model_class import NERmodel
from image_classification_model import CNNClassifier
from keras.models import load_model
import tensorflow as tf
class Pipeline_for_task_2():
    def __init__(self,image_classification_model_path,entity_recognition_model_path):
        self.classification_model = CNNClassifier(image_classification_model_path)
        self.ner_model=NERmodel(entity_recognition_model_path)


    def evaluate(self,image,string):

        animals = ['beetle', 'butterfly', 'cat', 'cow', 'dog', 'elephant', 'gorilla', 'hippo', 'lizard', 'monkey',
                   'mouse','panda', 'spider', 'tiger', 'zebra']  # animals on which the image classification model was trained

        img_class_pred=self.classification_model.predict(image)

        pred_animal=animals[img_class_pred[0]]

        token_id=self.ner_model.predict(string)
        token_pred=str.lower(token_id)

        print(f'Image predict: {pred_animal}.')
        print(f'\nToken predict: {token_pred}.\n')
        if pred_animal not in token_pred:
            return False
        else:
            return True

        print('a')


