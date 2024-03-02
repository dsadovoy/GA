from selection_methods import rank_selection, boltzmann_selection, tournament_selection
from crossover_methods import single_point_crossover
from mutation_methods import gaussian_mutation
from  lunar_lander  import LunarLander
from ga import GA
from in_population import InPopulation

crossover_prob = 0.5
min_crossover_prob = crossover_prob/5

mutation_rate = 0.1
mutation_std = 0.1

min_mutation_rate = mutation_rate/5
min_mutation_std = mutation_std/5


def fitness_method (chromosome):
    return env.get_score(chromosome)

def selection_method(fitness):
    # return tournament_selection(fitness)
    return rank_selection(fitness)
    # return boltzmann_selection(fitness, temperature)

def crossover_method(chromosome, crossover_prob):
    return single_point_crossover(chromosome, crossover_prob)

def mutation_method(chromosome, mutation_rate, mutation_std):
    return gaussian_mutation(chromosome, mutation_rate, mutation_std)


env = LunarLander()
elites_percent = 0.10
pop_size = 10
max_pop_size = pop_size*2
min_pop_size = pop_size/2
generations_num_max = 10
in_population = InPopulation(pop_size).generate_inpop()
dir  = 'C:/Users/Denis/Documents/London/New 2019/Final Project/ft/code/test/'
        # self.dir = '/content/drive/Othercomputers/My Laptop/London/New 2019/Final Project/ft/code/model/'
        # self.dir = 'model/'
        # self.dir = 'C:/Users/Denis/Documents/London/New 2019/Final Project/ft/code/model/'
ga_non_adapt = GA(dir, in_population, fitness_method, selection_method, crossover_method, mutation_method,  elites_percent, generations_num_max, min_pop_size, max_pop_size, min_crossover_prob, 
               min_mutation_rate, min_mutation_std, crossover_prob, mutation_rate, mutation_std, True)
ga_non_adapt.execute()
ga_non_adapt.save()
