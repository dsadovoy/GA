import tensorflow as tf 
from tensorflow.keras import models
from tensorflow.keras import layers

from chromosome import Chromosome

class InPopulation:
    def __init__(self, pop_size):     
      self.pop_size = pop_size
    
    def generate_inpop(self):
      in_population = []
      for i in range(self.pop_size):
         chromosome = Chromosome().get_weights_array()
         in_population.append(chromosome)
      return in_population


           
