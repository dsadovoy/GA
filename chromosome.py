import tensorflow as tf 
from tensorflow.keras import models
from tensorflow.keras import layers

import numpy as np

class Chromosome:
   def __init__(self):
    self.model = models.Sequential()
    self.model.add(layers.Input(8,))
    self.model.add(layers.Dense(256, activation = 'tanh'))
    # self.model.add(layers.Dense(256, activation = 'tanh'))
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