from ga import GA
import numpy as np
import matplotlib.pyplot as plt
from in_population import InPopulation

from selection_methods import rank_selection
from crossover_methods import single_point_crossover
from mutation_methods import gaussian_mutation
from  lunar_lander  import LunarLander


best_solution_200 = np.load("C:/Users/Denis/Documents/London/New 2019/Final Project/ft/code/model/best_solution200.npy")

env = LunarLander()
scores = []
for i in range(100): 
    scores.append(env.get_score(best_solution_200))

scores_mean = np.mean(scores)
print (scores_mean)
