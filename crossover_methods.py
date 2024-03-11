import numpy as np

def single_point_crossover(parents, crossover_prob):
    """ 
   Return child after single point crossover of two parents

   Parameters:
        parents: list of two parents
        crossover_prob: probability of crossover
   Return:
        Child after single point crossover of two parents
    """
    if np.random.random()<crossover_prob:
        parent_len=len(parents[0])
        point=np.random.choice(parent_len)
        child=np.concatenate((parents[0][:point],parents[1][point:]))
    else:
        child=parents [np.random.choice([0,1])]
    return child