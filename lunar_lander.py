import gym
import numpy as np
import random
import tensorflow as tf 
from tensorflow.keras import models
from tensorflow.keras import layers
from chromosome import Chromosome

class LunarLander:
    def __init__(self):
        
        self.env = gym.make("LunarLanderContinuous-v2")
        self.seed_count = 0
        # self.seed(self.seed_count)


    def seed(self, seed_int=0):
        random.seed(seed_int)
        np.random.seed(seed_int)
        self.env.seed(seed_int)
        self.env.action_space.seed(seed_int)
        self.seed_count+=1

    def get_score(self, flat_weights):
        chromosome = Chromosome()
        model = chromosome.model
        score = 0
        done = False
        weights = chromosome.restore_weights(flat_weights)
        model.set_weights(weights)
        
        # self.seed(self.seed_count)
        observation = self.env.reset()
          
        while not done:
            # self.env.render()
            observation = np.expand_dims(observation, axis=0)
            action = model.predict(observation, verbose=0)[0]
            observation, reward, done, info = self.env.step(action)
            score+=reward
        return score