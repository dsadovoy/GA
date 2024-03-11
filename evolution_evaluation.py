
import numpy as np
import matplotlib.pyplot as plt



def evolution_evaluation(dir):
    """ 
   Return average fitness and average best fitness across all generations
   Also create visualisations of them

   Parameters:
        dir: Full path to fitness history

   Return:
        List of average fitness and average best fitness across all generations
    """
    save_path = dir
    ga_history = np.load(dir + "fitness_history.npy")
    fit_mean = []
    fit_std = []
    for generation in ga_history:
        fit_mean.append(generation[2])
        fit_std.append(generation[3])

    fit_mean = np.array(fit_mean)
    fit_std = np.array(fit_std)
    plt.figure(figsize=(10,10))
    plt.plot(fit_mean)
    plt.fill_between(list(range(len(fit_mean))), fit_mean-fit_std, fit_mean+fit_std, alpha=0.4)
    plt.ylabel('Fitness mean')
    plt.xlabel('Number of generations')
    plt.title('Population fitness mean and standard deviation')
    plt.savefig(save_path+'/pop_fitness.png')
    plt.show()

    print("Average fitness over GA run: " + str(np.mean(fit_mean)))

    current_best=[]

    for generation in ga_history:
        current_best.append(generation[1])
        
    plt.figure(figsize=(10,10))
    plt.plot(current_best)
    plt.title('Best fitness of each generation')
    plt.xlabel('Number of generations')
    plt.ylabel('Fitness')
    plt.savefig(save_path+'/best_fitness.png')
    plt.show()

    print("Average best fitness over GA run: " + str(np.mean(current_best)))

    return [fit_mean, current_best]

def adapt_evolution_evaluation(dir):
    """ 
   Return list of adaptive parameters change for all generations for adaptive GA
   Also create visualisations of them

   Parameters:
        dir: Full path to adaptive parameters history

   Return:
        List of adaptive parameters change for all generations for adaptive GA
    """
    save_path = dir
    ga_history = np.load(dir + "adapt_history.npy")
    adapt_pop_size = []
    adapt_crossover_prob = []
    adapt_mutation_rate = []

    for generation in ga_history:
        adapt_pop_size.append(generation[1])
        adapt_crossover_prob.append(generation[2])
        adapt_mutation_rate.append(generation[3])

    plt.figure(figsize=(10,10))
    plt.plot(adapt_pop_size)
    plt.title('Population size change through GA run')
    plt.savefig(save_path+'/pop_adapt_history.png')
    plt.show()

    plt.figure(figsize=(10,10))
    plt.plot(adapt_crossover_prob, label = "Crossover probability")
    plt.plot(adapt_mutation_rate, label = "Mutation rate/std")
    plt.title('Crossover and mutation parameters change through GA run')
    plt.legend()
    plt.savefig(save_path+'/crs_mut_adapt_history.png')
    plt.show()

    
   
        
    print("Average population size over GA run: " + str(np.mean(adapt_pop_size)))
    print("Average crossover probability over GA run: " + str(np.mean(adapt_crossover_prob)))
    print("Average mutation rate/std over GA run: " + str(np.mean(adapt_mutation_rate)))

    return [adapt_pop_size, adapt_crossover_prob, adapt_mutation_rate]
