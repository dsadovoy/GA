import numpy as np

def rank_selection(fitness):
    """ 
   Return probabilities of parents selection for mating based on rank selection

   Parameters:
        fitness: list of parents' fitness
   Return:
        List of parents' probabilities
    """
    pop_size = len(fitness)
    parents_probs = np.zeros(pop_size)
    parents_probs_sorted = np.argsort(fitness)
    for i in range(pop_size):
        parents_probs[parents_probs_sorted[i]]+=2*(i+1)/(pop_size*(pop_size+1))
    return parents_probs

def tournament_selection(fitness, k=3):
    """ 
   Return probabilities of parents selection for mating based on deterministic tournament selection

   Parameters:
        fitness: list of parents' fitness
        k: number of competitors
   Return:
        List of parents' probabilities
    """
    pop_size = len(fitness)
    parents_probs = np.zeros(pop_size)
    for i in range(pop_size):
        competitors = np.random.choice(fitness, k, replace=False)
        competitors_sorted = np.sort(competitors)
        winner_ind = fitness.index(competitors_sorted[-1])
        parents_probs[winner_ind]+=1/pop_size
    return parents_probs

