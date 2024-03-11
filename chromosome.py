import tensorflow as tf 
from tensorflow.keras import models
from tensorflow.keras import layers

import numpy as np

class Chromosome:
   """ 
   A class that represents individual solution - neural network.

   Attributes:
        model (object): sequential Keras neural network
   Methods:
        get_weights_array(self): flatten Keras neural network to array
        restore_weights(self, flat_weights): restore flat array with weights to Keras neural network
   """
   def __init__(self):
    self.model = models.Sequential()
    self.model.add(layers.Input(8,))
    self.model.add(layers.Dense(256, activation = 'tanh'))
    self.model.add(layers.Dense(2, activation = 'tanh'))
    
   
   def get_weights_array(self):
    chromosome = np.empty(0)
    for layer in self.model.get_weights():
        chromosome = np.append(chromosome, layer)
    return chromosome
   
   def restore_weights(self, flat_weights):
    weights = []
    start = 0
    for layer in self.model.get_weights():
        increment = 1
        layer_shape = layer.shape
        for dim in layer_shape:
          increment*=dim
        weights.append(np.reshape(flat_weights[start:start+increment], layer_shape))
        start+=increment     
    return weights