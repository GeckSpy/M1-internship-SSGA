# SSGA band selection
# based on https://github.com/hszhaohs/SSGA

import numpy as np
import matplotlib.pyplot as plt
from EntropyRateSuperpixel import find_superpixel, complete_basic_similarity, create_img_with_borders, Superpixel


### Genetic operations
Indiv = np.ndarray[int]
def create_individual(B:int, b:int)->Indiv:
    """
    Create a random individual of lenght B with b 1
    """
    indiv = np.zeros(B, dtype=int)
    indiv[:b] = 1
    np.random.shuffle(indiv)
    return indiv


def create_population(B:int, b:int, NP:int)->list[Indiv]:
    """
    Create a populataion of NP random individual of lenght B with b 1
    """
    return [create_individual(B,b) for _ in range(NP)]


def cross_over(i1:Indiv, i2:Indiv, B:int, b:int)->tuple[Indiv, Indiv]:
    """
    Create two cross-overs based on i1 and i2
    """
    c1 = np.zeros(B, dtype=int)
    c2 = np.zeros(B, dtype=int)

    nb_1 = 0
    indices_differ = []
    for i in range(B):
        if i1[i]==1 and i2[i]==1:
            c1[i] = 1
            c2[i] = 1
            nb_1 += 1
        else:
            indices_differ.append(i)
    
    for i in np.random.choice(indices_differ, b-nb_1, replace=False):
        c1[i] = 1
    for i in np.random.choice(indices_differ, b-nb_1, replace=False):
        c2[i] = 1
    
    return c1, c2


def mutate(x:Indiv)->None:
    """
    Randomly mutate the individual x
    """
    id_0 = np.random.choice(np.where(x==0)[0])
    id_1 = np.random.choice(np.where(x==1)[0])
    x[id_0] = 1
    x[id_1] = 0




### Fitness function
def compute_SP_averages(data:np.ndarray, SP:Superpixel)->tuple[np.ndarray, np.ndarray]:
    """
    Compute the m_k and m (from paper)
    - m = mean vector of all superpixels
    - m_k = mean vector of the kth superpixel
    """
    n,m,B = data.shape
    SP_averages = np.zeros((len(SP), B))
    SP_average = np.zeros(B)

    for i,superpixel in enumerate(SP):
        for x,y in superpixel:
            SP_averages[i] += data[x,y]
        SP_average += SP_averages[i]
        SP_averages[i] /= len(SP[i])
    SP_average /= n*m

    return SP_averages, SP_average


def compute_Sbsp_Stsp_list(data:np.ndarray, SP:Superpixel)->tuple[np.ndarray, np.ndarray]:
    """
    Compute two arrays containing Sbsp and Stsp for each band
    """
    _,_,B = data.shape
    SP_averages, SP_average = compute_SP_averages(data, SP)

    Sbsp_list = np.zeros(B)
    Stsp_list = np.zeros(B)
    lens = np.array([len(SP[k]) for k in range(len(SP))])
    for b in range(B):
        Sbsp_list[b] = (lens*(SP_averages[:,b]-SP_average[b])**2).sum()
        for k in range(len(SP)):
            for i in range(len(SP[k])):
                Stsp_list[b] += (data[SP[k][i]][b] - SP_average[b])**2
    
    return Sbsp_list, Stsp_list




def fitness_function(bands:Indiv, Sbsp_list:np.ndarray, Stsp_list:np.ndarray)->float:
    # Lower value, better cost
    # Ensure non-negattive fitness
    Sbsp = (bands * Sbsp_list).sum()
    Stsp = (bands * Stsp_list).sum()
    if Stsp == 0:
        return 1
    else:
        return Sbsp/Stsp


def compute_fitness(pop:list[Indiv], Sbsp_list:np.ndarray, Stsp_list:np.ndarray)->list[float]:
    """
    Compute a list of fitness score corresponding to the fitness of each individual in the population
    """
    return [fitness_function(x, Sbsp_list, Stsp_list) for x in pop]



## Construct following generation
def ranking(pop:list[Indiv], fitness:list[float])->tuple[list[Indiv], list[float]]:
    """
    Sort the population and the fitness list according to the fitness values
    """
    sorted_pop, sorted_fitness = zip(*sorted(zip(pop, fitness), key=lambda x:-x[1]))
    return list(sorted_pop), list(sorted_fitness)


def sus(pop:list[Indiv], fitness:list[float], N:int)->list[Indiv]:
    """
    Return the population of kept individuals
    """
    # In original code: N = nb_to_selct = GGAP(=0.9) * Nind(=100)
    N = int(N)
    F = sum(fitness) #cumulative fitness
    P = F/N
    start = np.random.uniform(0, P)
    pointers = [start + i*P for i in range(N)]
    cumulative_fitness = np.cumsum(fitness)

    keep = []
    i = 0
    for p in pointers:
        while cumulative_fitness[i] < p:
            i += 1
        keep.append(pop[i])
    return keep




def cross_overs(pop:list[Indiv], B:int, b:int, Pc:float)->None:
    """
    Apply randomly some cross-overs
    - B: length of an individual
    - b: number of 1 in an individual
    - Pc: probability of cross-over
    """
    for i in range(0, len(pop), 2):
        if np.random.rand() < Pc and i+1<len(pop):
            pop[i], pop[i+1] = cross_over(pop[i], pop[i+1], B, b)


def mutations(pop:list[Indiv], Pm:float)->None:
    """
    Apply randomly some mutations
    - Pm: probability of mutations
    """
    for i in range(len(pop)):
        if np.random.rand() < Pm:
            mutate(pop[i])


def next_pop(pop:list[Indiv], B:int, Sbsp_list:np.ndarray, Stsp_list:np.ndarray, b:int, Pc:float, Pm:float, NP:int
             )->tuple[list[Indiv], list[float]]:
    """
    Compute the population of the next generation and their associated fitness function
    - B: length of an individual
    - b: number of 1 in an individual
    - Pc: probability of cross-over
    - Pm: probability of mutation
    - NP: length of a population
    """
    fitness = compute_fitness(pop, Sbsp_list, Stsp_list)
    pop, fitness = ranking(pop, fitness)
    pop = sus(pop, fitness, NP)
    cross_overs(pop, B, b, Pc)
    mutations(pop, Pm)
    return pop, fitness



# Main function
def SSGA(data:np.ndarray, b:int, K:int, NG:int, NP:int, Pc:float, Pm1:float, Pm2:float, show_plots=False, SP=None):
    """ 
    - data: set of images
    - b: number of bands to select
    - K: number of superpixel
    - GN: maximum number of generation
    - NP: population size
    - Pc: cross-over probability
    - Pm1: starting mutation probability
    - Pm2: ending mutation probability
    - show_plots: show the bordered image of superpixels
    """
    _,_,B = data.shape
    # Compute SuperPixxels and their mean vectors
    if SP==None:
        SP = find_superpixel(data, K, 0.5*8, complete_basic_similarity, True)
    bordered_img = create_img_with_borders(data, SP)
    Sbsp_list, Stsp_list = compute_Sbsp_Stsp_list(data, SP)

    if show_plots:
        plt.imshow(bordered_img)
        print(len(SP), "superpixels of size:")
        print([len(sp) for sp in SP])
        plt.show()

    # Initialisation of genetic algorithm
    pop = create_population(B, b, NP)
    current_generation = 0
    fitness = compute_fitness(pop, Sbsp_list, Stsp_list)
    while( current_generation < NG):
        #print(current_generation, fitness[0])
        Pm = Pm1 + (Pm2-Pm1) * current_generation/NG
        pop, fitness = next_pop(pop, B, Sbsp_list, Stsp_list, b, Pc, Pm, NP)
        current_generation += 1

    compute_fitness(pop, Sbsp_list, Stsp_list)
    pop, fitness = ranking(pop, fitness)
    return pop, fitness




def example():
    class Param:
        b=1
        K=20
        GN=10
        NP=10
        Pc=0.9
        Pm1=0.02
        Pm2=0.1

    data = plt.imread("images/low_flower.png")
    pop, fitness = SSGA(data, Param.b, Param.K, Param.GN, Param.NP, Param.Pc, Param.Pm1, Param.Pm2)
    print(pop[0], fitness[0])

#example()


