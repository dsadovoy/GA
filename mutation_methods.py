import numpy as np

def gaussian_mutation(child, mutation_rate, std):
    child_len = len(child)
    child = np.array(child)
    mutation_size = np.random.binomial(child_len, mutation_rate)
    mutation_idx = np.random.choice(child_len, replace=False, size=mutation_size)
    mutations = np.random.normal(scale=std, size=mutation_size)
    child[mutation_idx]+=mutations
    return list(child)