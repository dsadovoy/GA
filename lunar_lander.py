import gym
import numpy as np
from chromosome import Chromosome

class LunarLander:
    """ 
   A class that represents Lunar Lander gaming environment.

   Attributes:
        env (object): OpenAI gym Lunar Lander gaming environment
   Methods:
        get_score(self, flat_weights): get total score per episode in gaming environment
    """
    def __init__(self):
        
        self.env = gym.make("LunarLanderContinuous-v2")

    def get_score(self, flat_weights):
        chromosome = Chromosome()
        model = chromosome.model
        score = 0
        done = False
        weights = chromosome.restore_weights(flat_weights)
        model.set_weights(weights)
        
        observation = self.env.reset()
          
        while not done:
            # self.env.render()
            observation = np.expand_dims(observation, axis=0)
            action = model.predict(observation, verbose=0)[0]
            observation, reward, done, info = self.env.step(action)
            score+=reward
        return score