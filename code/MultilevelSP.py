import numpy as np
from sklearn.decomposition import PCA

from classes import MinHeap
from EntropyRateSuperpixel import find_superpixel, norm1_similarity



### classic clusters separability/variability metrics
def anovaFtest(clusters:list[list[np.ndarray]], dist=norm1_similarity)->float:
    # High = well separated
    K = len(clusters)
    sizes = [len(cluster) for cluster in clusters]
    n = np.sum(sizes)
    averages = np.array([np.average(cluster, axis=0) for cluster in clusters])
    average = np.average([ts for cluster in clusters for ts in cluster], axis=0)

    BGV = 0
    WGV = 0
    for k in range(K):
        BGV += sizes[k] * dist(averages[k], average)
        for ts in clusters[k]:
            WGV += dist(ts, averages[k])
    
    if WGV==0:
        return np.inf
    return (n-K)/(K-1) * BGV/WGV


from sklearn.metrics import davies_bouldin_score
def invertedDaviesBouldinIndex(clusters:list[list[np.ndarray]], dist=None)->float:
    # Lower value means better clustering then we invert
    labels = [i for i,cluster in enumerate(clusters) for _ in range(len(cluster))]
    X = [ts for cluster in clusters for ts in cluster]
    return 1/davies_bouldin_score(X, labels)


def DunnIndex(clusters:list[list[np.ndarray]], dist=norm1_similarity)->float:
    # High = well separated
    K = len(clusters)
    averages = np.array([np.average(cluster, axis=0) for cluster in clusters])

    minAveragesDist = dist(averages[0], averages[1])
    maxDiameter = 0
    for i in range(K):
        for j in range(i+1, K):
            minAveragesDist = min(dist(averages[i], averages[j]), minAveragesDist)

        diameter = 0
        for j in range(len(clusters[i])):
            for k in range(j+1, len(clusters[i])):
                diameter = max(diameter, dist(clusters[i][j], clusters[i][k]))
        maxDiameter = max(diameter, maxDiameter)
    
    return minAveragesDist/maxDiameter


def invertedXieBeniIndex(clusters:list[list[np.ndarray]], dist=norm1_similarity)->float:
    # low value better then we invert
    K = len(clusters)
    sizes = [len(cluster) for cluster in clusters]
    n = np.sum(sizes)
    averages = np.array([np.average(cluster, axis=0) for cluster in clusters])

    sum = 0
    minAveragesDist = dist(averages[0], averages[1])
    for k in range(K):
        for ts in clusters[k]:
            sum += dist(averages[k], ts)

        for j in range(k+1, K):
            minAveragesDist = min(dist(averages[k], averages[j]), minAveragesDist)
    return n*minAveragesDist/sum


### AnovaF-test-based Standard Deviation weighted variability function
def normalize(vec):
    arr = np.array(vec)
    mini = arr.min()
    maxi = arr.max()
    if mini==maxi:
        return arr/len(arr)
    return (arr-mini)/(maxi-mini)


def stdFtestnorm1(clusters, dist=(1,"")):
    # High = well separated
    coeff, _ = dist
    K = len(clusters)
    sizes = [len(cluster) for cluster in clusters]
    n = np.sum(sizes)
    averages = np.array([np.average(cluster, axis=0) for cluster in clusters])
    if dist[1]=="exp":
        stds = [normalize(np.exp(np.std(cluster, axis=0))) for cluster in clusters]
    else:
        stds = [normalize(np.std(cluster, axis=0)) for cluster in clusters]

    TS = [ts for cluster in clusters for ts in cluster]
    average = np.average(TS, axis=0)
    std = normalize(np.std(TS, axis=0))


    BGV = 0
    WGV = 0
    if coeff=="exp":
        for k in range(K):
            BGV += sizes[k] * (np.abs(averages[k]-average) * np.exp(-stds[k]*std)).sum()
            for ts in clusters[k]:
                WGV += (np.abs(ts-averages[k]) * np.exp(-stds[k]**2)).sum()
    else:
        for k in range(K):
            BGV += sizes[k] * (np.abs(averages[k]-average) * (1-stds[k]*std)**coeff).sum()
            for ts in clusters[k]:
                WGV += (np.abs(ts-averages[k]) * (1-stds[k]**2)**coeff).sum()
    
    if WGV==0:
        return np.inf
    return (n-K)/(K-1) * BGV/WGV



### Compute information for multileve-based algorithm
def compute_Ks(N:int, M:int, averageSPSize:int=5)->list[int]:
    Ksmax = int(N*M/averageSPSize)
    Ks = [20, 50]
    k = 100
    while k<Ksmax:
        Ks.append(k)
        k *=2
    Ks.sort()
    return Ks



def createMultilevelInfo(data:np.ndarray, Ks:list[int]):
    N,M = data.shape[0], data.shape[1]
    SPsDic = find_superpixel(data, Ks, lambda_coef="auto", simFun="norm1")
    associations = np.zeros((len(Ks), N, M), dtype=int)
    for level, K in enumerate(Ks):
        SPs = SPsDic[K]
        for k,SP in enumerate(SPs):
            for x,y in SP:
                associations[level][x,y] = k

    def getSP(level, idSP):
        return SPsDic[Ks[level]][idSP]

    getParent = [[None for _ in range(len(SPsDic[K]))] for K in Ks]
    for level in range(1,len(Ks)):
        for k, SP in enumerate(SPsDic[Ks[level]]):
            getParent[level][k] = associations[level-1][SP[0]]


    getChilds = [[[] for _ in range(len(SPsDic[K]))] for K in Ks]
    for level in range(1, len(Ks)):
        SPs = SPsDic[Ks[level]]
        for k, SP in enumerate(SPs):
            botId = getParent[level][k]
            getChilds[level-1][botId] += [k]

    return SPsDic, getSP, getParent, getChilds



### Main algorithm
def multilevelSPsegmentation(data, K, n_component=0, varFun=anovaFtest, dist=norm1_similarity, infos=None):
    N,M,_ = data.shape
    Ks = compute_Ks(N,M)
    if infos==None:
        infos = createMultilevelInfo(data, Ks)
    SPsDic, getSP, _, getChilds = infos

    def divide_comp_var(level, idSP, n_component=n_component):
        childsID = getChilds[level][idSP]
        if len(childsID)==0: return 0
        if len(childsID)==1: return np.inf

        childs = [getSP(level+1, id) for id in childsID]
        if n_component==0:
            clusters = [[data[coor] for coor in child] for child in childs]
            return varFun(clusters, dist=dist)
        
        clusters_id = [i for i,child in enumerate(childs) for _ in range(len(child))]
        TS = np.array([data[coor] for child in childs for coor in child])

        n_component = min(n_component, min(TS.shape))
        pca = PCA(n_components=n_component)
        coeffs = pca.fit_transform(TS)

        clusters = [[] for _ in range(len(childs))]
        for i in range(len(clusters_id)):
            clusters[clusters_id[i]].append(coeffs[i])
        clusters = [np.array(cluster) for cluster in clusters]
        return varFun(clusters, dist=dist)

    heap = MinHeap()
    for k in range(len(SPsDic[Ks[0]])):
        heap.insert((0, k), -divide_comp_var(0, k))

    while 0<len(heap.array)<K:
        elt,w = heap.pop()
        level, idSP = elt
        childsID = getChilds[level][idSP]
        if childsID==[]:
            heap.insert((level, idSP), 0)
            break
        for id in childsID:
            heap.insert((level+1, id), -divide_comp_var(level+1, id))

    return [getSP(pair.first[0], pair.first[1]) for pair in heap.array]
