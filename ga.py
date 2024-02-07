import numpy as np
import os
import time
# from google.colab import drive
class GA():
    def __init__(self, in_population, fitness_method, selection_method, crossover_method, mutation_method, elites_percent):
        self.population = in_population
        self.fitness_method = fitness_method
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.evol_rate_min = 0.01
        self.gens_evol_rate = 20
        self.generations_num_max = 225
        self.start_time = None
        self.dir = '/content/drive/Othercomputers/My Laptop/London/New 2019/Final Project/ft/code/model/'
        
        # self.dir = 'C:/Users/Denis/Documents/London/New 2019/Final Project/ft/code/model/'
        self.fitness_history = []
        self.best_fitness = None
        self.best_solution = None
        self.generations_num = 0
        self.elites_qty = int(round(len(self.population)*elites_percent,0))
        self.fitness = None 
#Computing fitness of solutions
    def compute_fitness(self):
        self.fitness= list(map(self.fitness_method, self.population))
        cur_best_fitness = np.max(self.fitness)
        if self.best_fitness is None:
            self.best_fitness = cur_best_fitness
            self.best_solution = self.population[np.argmax(self.fitness)]

        elif cur_best_fitness > self.best_fitness:
            self.best_fitness = cur_best_fitness
            self.best_solution = self.population[np.argmax(self.fitness)]
#Saving history of fitness
        self.fitness_history.append((self.generations_num, cur_best_fitness, 
                                  np.mean(self.fitness), np.std(self.fitness)))
#Generating population
    def generate_pop(self):
        pop_size = len(self.population)
        parents_probs = self.selection_method(self.fitness)
        children = []
#Impelementation of elitism
        elites_indx = list(np.argsort(parents_probs))[:self.elites_qty]

        for i in elites_indx:
            children.append(self.population[i])
        
        for i in range (pop_size - self.elites_qty):
            parents_mating_indx = np.random.choice(pop_size, size=2, replace=False, p=parents_probs)
            parents_mating = [self.population[parents_mating_indx[0]], self.population[parents_mating_indx[1]]]
            child = self.crossover_method(parents_mating)
            child = self.mutation_method(child)
            children.append(child)
        self.population = children
   
    def is_evol_positive(self, value, gens_evol_rate, evol_rate_min):
        mean_fitness_pop = []

        for data in self.fitness_history[-gens_evol_rate:]:               
            mean_fitness_pop.append(data[2])      
        mean_fitness_gens_evol_rate = np.mean(mean_fitness_pop)
        evol_rate_diff = value - mean_fitness_gens_evol_rate
        return evol_rate_diff < evol_rate_min

        
#Executing run of GA
    def execute(self):
        
        self.start_time  = time.time()
        if self.fitness == None:
            self.compute_fitness()
            self.generations_num+=1
        
        for i in range(self.generations_num_max):
            self.generate_pop()
            self.compute_fitness()

#Termination condition
            if i >self.gens_evol_rate and self.is_evol_positive(self.best_fitness, self.gens_evol_rate, self.evol_rate_min):
                self.save(self.dir)
                break
            np.save(self.dir + 'best_solution'+str(i+1), self.best_solution)
            self.generations_num+=1
        self.save(self.dir)
#Saving results
    def save(self, dir):

        # if not os.path.exists(dir+'/model'):
        #     os.makedirs(dir+'/model')
        # np.save(dir+'/model/population', self.population)
        # np.save(dir+'/model/fitness', self.fitness)
        # np.save(dir+'/model/generation', self.generations_num)
        # np.save(dir+'/model/fitness_history', self.fitness_history)
        # np.save(dir+'/model/best_fitness', self.best_fitness)
        # np.save(dir+'/model/best_solution', self.best_solution)
        # print('Model saved')
        # try:
        # # Check if Google Drive is still connected
        #     os.listdir('/content/drive')
        # except:
        # # If not, reconnect it
        
        #     drive.mount('/content/drive', force_remount=True)
        np.save(dir + 'fitness_history', self.fitness_history)
        end_time = time.time()
        dur_hours = (end_time - self.start_time)/3600
        f = open(dir + 'dur_hours.txt', 'w')
        f.write(str(dur_hours))
        f.close()