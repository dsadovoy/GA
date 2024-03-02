from ga import GA
import numpy as np
import matplotlib.pyplot as plt
from in_population import InPopulation

from selection_methods import rank_selection
from crossover_methods import single_point_crossover
from mutation_methods import gaussian_mutation
from  lunar_lander  import LunarLander

# ga_history = np.load("C:/Users/Denis/Documents/London/New 2019/Final Project/ft/code/model_1hl_250_el01_cr05_seed/fitness_history.npy")
# ga_history = np.load("C:/Users/Denis/Documents/London/New 2019/Final Project/ft/code/model_1hl_225_el01_cr05tour/fitness_history.npy")

# current_best=[]

# for el in ga_history:
#     current_best.append(el[1])

# best_id = np.argmax(current_best)+1
# best_solution= np.load("C:/Users/Denis/Documents/London/New 2019/Final Project/ft/code/model_1hl_225_el01_cr05tour/best_solution"+str(best_id) +".npy")
    
best_solution= np.load("C:/Users/Denis/Documents/London/New 2019/Final Project/ft/code/model5/best_solution250.npy")

env = LunarLander()
scores = []
for i in range(100): 
    scores.append(env.get_score(best_solution))

scores_mean = np.mean(scores)
print (scores_mean)
