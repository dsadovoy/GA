import numpy as np

def single_point_crossover(parents, crossover_prob):
    if np.random.random()<crossover_prob:
        parent_len=len(parents[0])
        point=np.random.choice(parent_len)
        child=np.concatenate((parents[0][:point],parents[1][point:]))
    else:
        child=parents [np.random.choice([0,1])]
    return child