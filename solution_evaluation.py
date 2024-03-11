
import numpy as np
from  lunar_lander  import LunarLander


def solution_evaluation (dir, episodes_num=100):
    """ 
   Return scores of evaluated solution for defined number of episodes

   Parameters:
        dir: Full path to evaluated solution
        episodes_num: number of episodes for evaluation
   Return:
        List of achieved score and print average score
    """
    eval_solution= np.load(dir)

    env = LunarLander()
    scores = []
    for i in range(episodes_num): 
        scores.append(env.get_score(eval_solution))

    scores_mean = np.mean(scores)
    print ("Average score per 100 episodes: " + str(scores_mean))
    np.save(dir + "scores", scores)
    return scores
    


