import numpy as np

def rank_selection(fitness):
    pop_size = len(fitness)
    parents_probs = np.zeros(pop_size)
    parents_probs_sorted = np.argsort(fitness)
    for i in range(pop_size):
        parents_probs[parents_probs_sorted[i]]+=2*(i+1)/(pop_size*(pop_size+1))
    return parents_probs