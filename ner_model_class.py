import numpy as np
import tensorflow as tf
from transformers import TFBertForTokenClassification, AutoTokenizer


class NERmodel():
    def __init__(self,path_to_model):
        self.model=TFBertForTokenClassification.from_pretrained(path_to_model)
        self.tokenizer=AutoTokenizer.from_pretrained(path_to_model)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
        #
        # self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print('Model is loaded.')
    def train(self,train_dataset):

        pass
    def predict(self,test_dataset):

        inputs = self.tokenizer(test_dataset, return_tensors="tf")
        output=self.model(**inputs)
        logits=np.argmax(output.logits,axis=-1)[0]
        max_index=np.argmax(logits)
        # token_id=
        tokens=self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        return tokens[max_index]
