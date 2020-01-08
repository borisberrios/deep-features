import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor
import os

class CNN:

    def __init__(self, base_dir_model):

        self.cnn_predictor = predictor.from_saved_model(base_dir_model)
        self.shape = (128,128)
        self.mean = np.fromfile(os.path.join(base_dir_model,'mean.dat'), dtype=np.float32)
        self.mean = np.reshape(self.mean, (128,128, 1))

    def feature(self, image):
      #print(image.dtype, self.mean.shape, self.mean.dtype)
      #image = image.astype(np.float32)
      normalized_image = image - self.mean
      predictions = self.cnn_predictor({"input":[normalized_image]})
      feature = predictions['deep_features']
      return feature

    def features(self, images):
      normalized_images = images - self.mean
      predictions = self.cnn_predictor({"input":normalized_images})
      features = predictions['deep_features']
      return features

    def prediction(self, image):
      normalized_image = image - self.mean
      predictions = self.cnn_predictor({"input":[normalized_image]})
      prob = np.squeeze(predictions['predicted_probabilities'])
      return np.argmax(prob)

    def predictions(self, images):
      normalized_images = images - self.mean
      predictions = self.cnn_predictor({"input":normalized_images})
      prob = np.squeeze(predictions['predicted_probabilities'])
      return np.argmax(prob, axis = 1)
