import numpy as np
import matplotlib.pyplot as plt

from MyDataset import standardize_data
from EntropyRateSuperpixel import find_superpixel

gamma = 0.15

def computePs(K, N, M):
    return int(N*M/K * 1/np.log(np.log(N*M/K)) * gamma)


def dichotomies_search(data, K, mini, maxi, similarity_function, Ps, print_info=False):
    def aux(i,j, SP):
        coeff = int((i+j)/2)
        if print_info:
            print("finding lambda coeff:", coeff)
        if i>=j:
            return SP, max(coeff,1)
        
        SP = find_superpixel(data, K, coeff, similarity_function)
        minSPsize = min([len(l) for l in SP])
        if minSPsize==Ps:
            return SP, coeff
        elif minSPsize<Ps:
            return aux(coeff+1, j, SP)
        else:
            return aux(i, coeff-1, SP)
        
    return aux(mini, maxi, None)


def findByCroping(data, simFun, nbCroping, n, m):
    # Didn't performmed well
    N,M,B = data.shape
    sum_lambda = 0

    K = 100
    a = int(20*K/100)
    b = int(60*K/100)
    for _ in range(nbCroping):
        xStart = np.random.randint(0, N-1-n)
        yStart = np.random.randint(0, M-1-m)
        trainData = standardize_data(data)[xStart:xStart+n, yStart:yStart+m, :]
        plt.imshow(trainData[:,:,1])
        plt.show()
        Ps = computePs(K, N, M)
        _, coeff = dichotomies_search(trainData, K, a, b, simFun, Ps, True)
        print(coeff)
        sum_lambda += coeff
    
    return sum_lambda/nbCroping



def getLambdaAverage(K,N,M):
    return 0.38 * gamma* K * np.log(N*M*K)**0.668

def getLambdaNorm2(K,N,M):
    return 1.214 * gamma* K * np.log(N*M*K)**0.444

def getLambdaNorm1(K,N,M):
    return 0.176* gamma* K * np.log(N*M)**1.147