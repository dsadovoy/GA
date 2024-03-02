import numpy as np
import os
import time
# from google.colab import drive
class GA():
    def __init__(self, dir, in_population, fitness_method, selection_method, crossover_method, mutation_method, elites_percent, generations_num_max, 
                 min_pop_size = 30, max_pop_size = 60,
                 min_crossover_prob = 0.05, min_mutation_rate = 0.05, min_mutation_std= 0.05, crossover_prob =0.5, mutation_rate=0.1, mutation_std=0.1,adapt = False):
        self.population = in_population
        self.fitness_method = fitness_method
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.crossover_prob = crossover_prob
        self.mutation_method = mutation_method
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.min_crossover_prob = min_crossover_prob
        self.min_mutation_rate = min_mutation_rate
        self.min_mutation_std= min_mutation_std
        self.adapt = adapt
        self.adapt_crossover_prob = crossover_prob
        self.adapt_mutation_rate= mutation_rate
        self.adapt_mutation_std = mutation_std
        self.adapt_history = []
        self.pop_size = len(self.population)
        self.min_pop_size = min_pop_size
        self.max_pop_size = max_pop_size
        self.evol_rate_min = 0.01
        self.gens_evol_rate = 30
        self.generations_num_max = generations_num_max
        self.start_time = None
        self.dir = dir

      
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
        # pop_size = len(self.population)
        parents_probs = self.selection_method(self.fitness)
        # parents_probs = self.selection_method(self.fitness, max(1 - 0.02*(self.generations_num+1), 0.1))
        children = []
#Impelementation of elitism
        elites_indx = list(np.argsort(parents_probs))[-self.elites_qty:]

        for i in elites_indx:
            children.append(self.population[i])
        if self.adapt and self.generations_num >self.gens_evol_rate:
            is_evol_positive = self.is_evol_positive( self.fitness_history[self.generations_num -1][2],self.gens_evol_rate, self.evol_rate_min)
        if self.adapt and self.generations_num >self.gens_evol_rate and is_evol_positive:
            self.adapt_crossover_prob = max (self.adapt_crossover_prob * 0.99, self.min_crossover_prob)
            self.adapt_mutation_rate = max (self.adapt_mutation_rate * 0.99, self.min_mutation_rate)
            self.adapt_mutation_std = max (self.adapt_mutation_std * 0.99, self.min_mutation_std)
        elif self.adapt and self.generations_num >self.gens_evol_rate and not is_evol_positive:
            self.adapt_crossover_prob = min (self.adapt_crossover_prob * 1.1, 1)
            self.adapt_mutation_rate = min (self.adapt_mutation_rate * 1.1, 1)
            self.adapt_mutation_std = min (self.adapt_mutation_std * 1.1, 1)

        for i in range (self.pop_size  - self.elites_qty):
            parents_mating_indx = np.random.choice(self.pop_size , size=2, replace=False, p=parents_probs)
            parents_mating = [self.population[parents_mating_indx[0]], self.population[parents_mating_indx[1]]]
            child = self.crossover_method(parents_mating, self.adapt_crossover_prob)
            child = self.mutation_method(child, self.adapt_mutation_rate, self.adapt_mutation_std)
            children.append(child)
        if self.adapt and self.generations_num >self.gens_evol_rate and  is_evol_positive and self.pop_size > self.min_pop_size:
            children.pop()
            self.pop_size-=1
        elif self.adapt and self.generations_num >self.gens_evol_rate and not is_evol_positive and self.pop_size < self.max_pop_size:
            children.append(self.population[list(np.argsort(parents_probs))[-self.elites_qty-1]])
            self.pop_size+=1
        if self.adapt:
            self.adapt_history.append((self.generations_num, self.pop_size, self.adapt_crossover_prob, self.adapt_mutation_rate, self.adapt_mutation_std))
        self.population = children

   
    def is_evol_positive(self, value, gens_evol_rate, evol_rate_min):
        mean_fitness_pop = []

        for data in self.fitness_history[-gens_evol_rate-1:-1]:               
            mean_fitness_pop.append(data[2])      
        mean_fitness_gens_evol_rate = np.mean(mean_fitness_pop)
        evol_rate_diff = value - mean_fitness_gens_evol_rate
        return evol_rate_diff > evol_rate_min

        
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
            if i >self.gens_evol_rate and not self.is_evol_positive(self.best_fitness, self.gens_evol_rate, self.evol_rate_min):
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
        if self.adapt:
            np.save(dir + 'adapt_history', self.adapt_history)
        end_time = time.time()
        dur_hours = (end_time - self.start_time)/3600
        f = open(dir + 'dur_hours.txt', 'w')
        f.write(str(dur_hours))
        f.close()