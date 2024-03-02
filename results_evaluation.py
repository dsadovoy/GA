from ga import GA
import numpy as np
import matplotlib.pyplot as plt
from in_population import InPopulation

from selection_methods import rank_selection
from crossover_methods import single_point_crossover
from mutation_methods import gaussian_mutation
from  lunar_lander  import LunarLander


# def fitness_method (chromosome):
#     return env.get_score(chromosome)

# def selection_method(fitness):

#     return rank_selection(fitness)

# def crossover_method(chromosome):
#     return single_point_crossover(chromosome, crossover_prob=0.5)

# def mutation_method(chromosome):
#     return gaussian_mutation(chromosome, mutation_rate=0.1, std=0.1)



# def load_ga(dir, fitness_fn, mutation_fn, crossover_fn,
#                  selection_fn, elites_percent):
#     '''Returns the loaded GeneticAlg instance'''
#     loaded=GA([], fitness_fn, mutation_fn, crossover_fn,
#                  selection_fn, elites_percent)
#     loaded.population=np.load(dir+'\\model\\population.npy' ).tolist()
#     loaded.fitness=np.load(dir+'\\model\\fitness.npy' ).tolist()
#     loaded.generations_num=np.load(dir+'\\model\\generation.npy')
#     loaded.fitness_history=np.load(dir+'\\model\\fitness_history.npy').tolist()
#     loaded.best_fitness=np.load(dir+'\\model\\best_fitness.npy')
#     loaded.best_solution=np.load(dir+'\\model\\best_solution.npy')
    
#     return loaded

# env = LunarLander()
# elites_percent = 0.15
# save_path='C:\\Users\\Denis\\Documents\\London\\New 2019\\Final Project\\GA stats'
# ga=load_ga(save_path, fitness_method, mutation_method, crossover_method, selection_method, elites_percent)

# fit_mean=[]
# fit_std=[]
# for el in ga.fitness_history:
#     fit_mean.append(el[2])
#     fit_std.append(el[3])

# fit_mean=np.array(fit_mean)
# fit_std=np.array(fit_std)
# plt.figure(figsize=(10,10))
# plt.plot(fit_mean)
# plt.fill_between(list(range(len(fit_mean))), fit_mean-fit_std, fit_mean+fit_std, alpha=0.4)
# plt.savefig(save_path+'/pop_fitness.png')
# plt.ylabel('Fitness mean')
# plt.xlabel('Number of generations')
# plt.title('Population fitness mean and standard deviation')
# plt.show()

save_path="C:/Users/Denis/Documents/London/New 2019/Final Project/ft/code/model5/"
ga_history = np.load("C:/Users/Denis/Documents/London/New 2019/Final Project/ft/code/model5/fitness_history.npy")
fit_mean=[]
fit_std=[]
for el in ga_history:
    fit_mean.append(el[2])
    fit_std.append(el[3])

fit_mean=np.array(fit_mean)
fit_std=np.array(fit_std)
plt.figure(figsize=(10,10))
plt.plot(fit_mean)
plt.fill_between(list(range(len(fit_mean))), fit_mean-fit_std, fit_mean+fit_std, alpha=0.4)
plt.savefig(save_path+'/pop_fitness.png')
plt.ylabel('Fitness mean')
plt.xlabel('Number of generations')
plt.title('Population fitness mean and standard deviation')
plt.show()

current_best=[]

for el in ga_history:
    current_best.append(el[1])
    
plt.figure(figsize=(10,10))
plt.plot(current_best)
plt.savefig(save_path+'/best_fitness.png')
plt.title('Best fitness of each generation')
plt.xlabel('Number of generations')
plt.ylabel('Fitness')
plt.show()

# histadapt = np.load("C:/Users/Denis/Documents/London/New 2019/Final Project/ft/code/model/adapt_history.npy")

# plt.figure(figsize=(10,10))
# plt.plot(histadapt[0])
# plt.plot(histadapt[1])
# plt.plot(histadapt[2])
# plt.plot(histadapt[3])
# plt.savefig(save_path+'/histadapt.png')
# plt.title('Adaptation of parameters')
# plt.xlabel('Number of generations')
# plt.ylabel('Parameters')
# plt.show()