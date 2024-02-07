from selection_methods import rank_selection
from crossover_methods import single_point_crossover
from mutation_methods import gaussian_mutation
from  lunar_lander  import LunarLander
from ga import GA
from in_population import InPopulation



def fitness_method (chromosome):
    return env.get_score(chromosome)

def selection_method(fitness):

    return rank_selection(fitness)

def crossover_method(chromosome):
    return single_point_crossover(chromosome, crossover_prob=0.5)

def mutation_method(chromosome):
    return gaussian_mutation(chromosome, mutation_rate=0.1, std=0.1)


env = LunarLander()
elites_percent = 0.15
pop_size = 50
in_population = InPopulation(pop_size).generate_inpop()

ga_non_adapt = GA(in_population, fitness_method, selection_method, crossover_method, mutation_method, elites_percent)
ga_non_adapt.execute()
ga_non_adapt.save()
