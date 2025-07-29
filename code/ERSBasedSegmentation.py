import time
import numpy as np
from sklearn.decomposition import PCA

from classes import MinHeap
from EntropyRateSuperpixel import find_superpixel, norm1_similarity, find_borders


### Toolbox function
def normalize(vec):
    arr = np.array(vec)
    mini = arr.min()
    maxi = arr.max()
    if mini==maxi:
        return arr/len(arr)
    return (arr-mini)/(maxi-mini)


def compute_medoid(group:list, dist=norm1_similarity):
    n,_ = group.shape
    distances = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i!=j:
                distances[i] += dist(group[i], group[j])
    metroid_index = np.argmin(distances)
    return group[metroid_index]


def center_distances(SP):
    def dist_squared(c1, c2):
        return (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2

    center = np.average(SP, axis=0)
    return np.array([dist_squared(coor, center) for coor in SP])


def compute_averages(clusters, dtc=None):
    if type(dtc)==type(None):
        return np.array([np.average(cluster, axis=0) for cluster in clusters])
    else:
        idtc = [np.zeros(len(d)) for d in dtc]
        for i in range(len(dtc)):
            x = np.max(dtc[i])-dtc[i]
            norm = np.linalg.norm(x)
            if norm==0:
                idtc[i] = [1/len(dtc[i]) for _ in range(len(dtc[i]))]
            else:
                idtc[i] = x/norm

        averages = [0 for _ in range(len(clusters))]
        for i in range(len(clusters)):
            sum = 0
            for j in range(len(clusters[i])):
                sum += clusters[i][j] * idtc[i][j]
            averages[i] = sum
        return averages


def compute_false_greyscale_img(data:np.ndarray):
    N,M,B = data.shape
    pca = PCA(n_components=3)
    pca.fit_transform([np.reshape(data[:,:,i], (N*M)) for i in range(B)])
    pca_data = np.array([np.reshape(pca.components_[i], (N, M)) for i in range(3)])
    return np.array([[[pca_data[b,i,j] for b in range(3)] for j in range(M)] for i in range(N)])



### classic clusters separability/variability metrics
def anovaFtest(clusters:list[list[np.ndarray]],
               averages:list[np.ndarray] = None,
               dist=norm1_similarity,
               dtc=None)->float:
    # High = well separated
    K = len(clusters)
    sizes = [len(cluster) for cluster in clusters]
    n = np.sum(sizes)
    if type(averages)==type(None):
        averages = compute_averages(clusters, dtc=dtc)
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
def invertedDaviesBouldinIndex(clusters:list[list[np.ndarray]], dist=None, dtc=None)->float:
    # Lower value means better clustering then we invert
    labels = [i for i,cluster in enumerate(clusters) for _ in range(len(cluster))]
    X = [ts for cluster in clusters for ts in cluster]
    return 1/davies_bouldin_score(X, labels)


def DunnIndex(clusters:list[list[np.ndarray]], dist=norm1_similarity, dtc=None)->float:
    # High = well separated
    K = len(clusters)
    averages = compute_averages(clusters, dtc=dtc)

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


def invertedXieBeniIndex(clusters:list[list[np.ndarray]], dist=norm1_similarity, dtc=None)->float:
    # low value better then we invert
    K = len(clusters)
    sizes = [len(cluster) for cluster in clusters]
    n = np.sum(sizes)
    averages = compute_averages(clusters, dtc=dtc)

    sum = 0
    minAveragesDist = dist(averages[0], averages[1])
    for k in range(K):
        for ts in clusters[k]:
            sum += dist(averages[k], ts)

        for j in range(k+1, K):
            minAveragesDist = min(dist(averages[k], averages[j]), minAveragesDist)
    return n*minAveragesDist/sum


def AverageDist(components:list[list[np.ndarray]], dist=norm1_similarity, dtc=None)->float:
    if len(components)<=1: return 0
    distances = 0
    count = 0
    for i in range(len(components)):
        comp1 = components[i]
        for j in range(i+1, len(components)):
            comp2 = components[j]

            n,_ = comp1.shape
            m,_ = comp2.shape
            distance_i = 0
            for a in range(n):
                for b in range(m):
                    distance_i += dist(comp1[a], comp2[b])
            count += 1
            distances += distance_i/(n*m)
    return distances/count


def stdFtestnorm1(clusters:list, dist=(1,""), dtc=None, averages=None):
    # High = well separated
    coeff, _ = dist
    K = len(clusters)
    sizes = [len(cluster) for cluster in clusters]
    n = np.sum(sizes)
    if type(averages)==type(None):
        averages = compute_averages(clusters, dtc=dtc)

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






### Merged-Based Neighborhood algorithm
from scipy.optimize import root_scalar
def computeKor(N:int, M:int, n_component:int=0,
               P_avg:float=35, gamma:float=0.15)->int:
    if n_component==0:
        return int(N*M/P_avg)

    def f(x):
        return x * np.log(np.log(x))

    def f_inverse(y):
        if y <= 0:
            raise ValueError("f(x) = x log(log(x)) is only defined for x > e")
        
        def equation(x):
            return f(x) - y
        
        result = root_scalar(equation, bracket=[np.e + 1e-5, 1e10], method='brentq')
        return result.root
    
    choosen_max = max(P_avg, f_inverse(n_component/gamma))
    return int(N*M/choosen_max)


def computeMergeBasedInfo(data:np.ndarray, n_component:int=0):
    N,M = data.shape[0], data.shape[1]
    K_or = computeKor(N,M, n_component=n_component)

    SPs_or = find_superpixel(data, K_or, lambda_coef="auto", simFun="norm1")
    pixelToSP = np.zeros((N,M), dtype=int)
    for k,SP in enumerate(SPs_or):
        for x,y in SP:
            pixelToSP[x,y] = k

    neighboors = [set() for _ in range(len(SPs_or))]
    borders = find_borders(SPs_or, (N,M), exterior=True)
    for k1 in range(len(borders)):
        for x,y in borders[k1]:
            k2 = pixelToSP[x,y]
            neighboors[k1].add(k2)
    return SPs_or, neighboors


def merge_SPs(SPs_or :list[list[tuple[int,int]]],
              neighboors_or :list[set[int]],
              trainData :np.ndarray,
              K :int,
              n_component :int =0,
              varFun=anovaFtest,
              dist =norm1_similarity,
              compare_comp :bool =False,
              weighted_avg :bool =False,
              starting_time:float = None):
    
    Ks = [K] if type(K)==int else [k for k in K]
    Ks.sort(key=lambda x:-x)

    def insert_sorted(l, elt):
        _,_,w = elt
        left, right = 0, len(l)
        while left < right:
            mid = (left+right)//2
            if l[mid][2]<w:
                left = mid+1
            else:
                right = mid
        l.insert(left, elt)


    if starting_time!=None:
        dic_time = {}
    SPs = [SP.copy() for SP in SPs_or]
    neighboors:list[set] = [neighboor.copy() for neighboor in neighboors_or]
    TSs = [[trainData[coor] for coor in SP] for SP in SPs]
    averages = compute_averages(TSs, dtc=None)


    def simFun(k1, k2, n_component=n_component, compare_comp=compare_comp):
        dtc = [center_distances(SP) for SP in [SPs[k1], SPs[k2]]] if weighted_avg else None
        if n_component==0:
            return varFun([TSs[k1],TSs[k2]], dist=dist, dtc=dtc, averages=[averages[k1],averages[k2]])
        if compare_comp:
            clusters = []
            for k in [k1, k2]:
                TS = np.array(TSs[k])
                n_ = min(n_component, min(TS.shape))
                pca = PCA(n_components=n_)
                try:
                    coeffs = pca.fit_transform(TS)
                except:
                    TS -= TS.mean(axis=0)
                    TS /= TS.std(axis=0) + 1e-8
                    coeffs = pca.fit_transform(TS)
                clusters.append(pca.components_ + pca.mean_)
            return varFun(clusters, dist=dist, dtc=dtc)
            
        TS = np.array(TSs[k1] + TSs[k2])
        n_ = min(n_component, min(TS.shape))
        pca = PCA(n_components=n_)
        try:
            coeffs = pca.fit_transform(TS)
        except:
            TS -= TS.mean(axis=0)
            TS /= TS.std(axis=0) + 1e-8
            coeffs = pca.fit_transform(TS)

        clusters = [[], []]
        for i in range(len(SPs[k1])):
            clusters[0].append(coeffs[i])
        for i in range(len(SPs[k2])):
            clusters[1].append(coeffs[i+len(SPs[k1])])
        clusters = [np.array(cluster) for cluster in clusters]
        return varFun(clusters, dist=dist, dtc=dtc)
    

    nb_cc = len(SPs)
    existing = [True for _ in range(nb_cc)]
    edges = [(u, v, simFun(u, v)) for u in range(nb_cc)
                for v in neighboors[u] if u<v]
    edges.sort(key=lambda x:x[2])
    
    SPsDic = {K:[] for K in Ks}
    for K in Ks:
        while nb_cc > K and len(edges)>0:
            k1,k2,_ = edges.pop(0)
            if existing[k1] and existing[k2]:
                existing[k2] = False
                SPs[k1] += SPs[k2]
                TSs[k1] += TSs[k2]
                averages[k1] = np.average(TSs[k1], axis=0)
                
                SPs[k2] = None
                neighboors[k1] = neighboors[k1].union(neighboors[k2])
                neighboors[k2] = None

                edges = [(u,v,w) for u,v,w in edges if u!=k1 and v!=k2 and u!=k2 and v!=k1 and existing[u] and existing[v]]
                for v in neighboors[k1]:
                    if v!=k1 and existing[v]:
                        insert_sorted(edges, (k1,v, simFun(k1, v)))
                
                nb_cc -=1
        SPsDic[K] = [[coor for coor in SP] for i,SP in enumerate(SPs) if existing[i]]
        if starting_time!=None:
            dic_time[K] = time.time() - starting_time

    res = SPsDic[Ks[0]] if len(Ks)==1 else SPsDic
    if starting_time!=None:
        return res, dic_time
    else:
        return res
    

def mergedBasedSegmentation(data :np.ndarray, K :int,
                            n_component :int=0,
                            usedVarFun=anovaFtest,
                            dist=norm1_similarity,
                            compare_comp=False,
                            infos=None,
                            weighted_avg :bool=False,
                            info_time:bool = False):
    
    starting_time = time.time() if info_time else None
    if n_component==0 and compare_comp:
        raise ValueError("Cannot compare PCA component for <=0 components")
    if infos==None:
        infos = computeMergeBasedInfo(data, n_component=n_component)
    SPs_or, neighboors = infos

    return merge_SPs(SPs_or, neighboors, data, K,
               n_component=n_component, varFun=usedVarFun,
               compare_comp=compare_comp, dist=dist, weighted_avg=weighted_avg,
               starting_time=starting_time)


### Multilevel Algo
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


def multilevelSPsegmentation(data :np.ndarray, K:int,
                             n_component :int=0,
                             varFun =anovaFtest,
                             dist =norm1_similarity,
                             infos =None,
                             compare_comp :bool=False,
                             weighted_avg :bool=False,
                             info_time :bool = False):
    if info_time:
        starting_time = time.time()
        dic_time = {}
    if n_component==0 and compare_comp:
        raise ValueError("Cannot compare PCA component for <=0 components")

    N,M,_ = data.shape
    Ks_or = compute_Ks(N,M)
    if infos==None:
        infos = createMultilevelInfo(data, Ks_or)
    SPsDic, getSP, _, getChilds = infos

    Ks = [K] if type(K)==int else [k for k in K]
    Ks.sort(key=lambda x:x)

    def divide_comp_var(level, idSP, n_component=n_component, compare_comp=compare_comp, weighted_avg=weighted_avg):
        childsID = getChilds[level][idSP]
        if len(childsID)==0: return 0
        if len(childsID)==1: return np.inf

        childs = [getSP(level+1, id) for id in childsID]
        dtc = [center_distances(SP) for SP in childs] if weighted_avg else None
        if n_component==0:
            clusters = [[data[coor] for coor in child] for child in childs]
            return varFun(clusters, dist=dist, dtc=dtc)
        
        if compare_comp:
            clusters = []
            for group in childs:
                TS = np.array([data[coor] for coor in group])
                n_ = min(n_component, min(TS.shape))
                pca = PCA(n_components=n_)
                try:
                    coeffs = pca.fit_transform(TS)
                except:
                    TS -= TS.mean(axis=0)
                    TS /= TS.std(axis=0) + 1e-8
                    coeffs = pca.fit_transform(TS)
                clusters.append(pca.components_ + pca.mean_)
            return varFun(clusters, dist=dist, dtc=dtc)

        clusters_id = [i for i,child in enumerate(childs) for _ in range(len(child))]
        TS = np.array([data[coor] for child in childs for coor in child])

        n_ = min(n_component, min(TS.shape))
        pca = PCA(n_components=n_)
        try:
            coeffs = pca.fit_transform(TS)
        except:
            TS -= TS.mean(axis=0)
            TS /= TS.std(axis=0) + 1e-8
            coeffs = pca.fit_transform(TS)

        clusters = [[] for _ in range(len(childs))]
        for i in range(len(clusters_id)):
            clusters[clusters_id[i]].append(coeffs[i])
        clusters = [np.array(cluster) for cluster in clusters]
        return varFun(clusters, dist=dist, dtc=dtc)

    heap = MinHeap()
    for k in range(len(SPsDic[Ks_or[0]])):
        heap.insert((0, k), -divide_comp_var(0, k))

    SPsDic = {}
    for K in Ks:
        while 0<len(heap.array)<K:
            elt,w = heap.pop()
            level, idSP = elt
            childsID = getChilds[level][idSP]
            if childsID==[]:
                heap.insert((level, idSP), 0)
                break
            for id in childsID:
                heap.insert((level+1, id), -divide_comp_var(level+1, id))
        
        SPsDic[K] = [[coor for coor in getSP(pair.first[0], pair.first[1])] for pair in heap.array]
        if info_time:
            dic_time[K] = time.time() - starting_time

    res = SPsDic[Ks[0]] if len(Ks)==1 else SPsDic
    if info_time:
        return res, dic_time
    else:
        return res



### Merge-SP algorithm
def globalSPsMerge(data :np.ndarray,
                   SPs: list[list[tuple[int,int]]],
                   K :int, 
                   n_component :int=0, 
                   usedVarFun=anovaFtest,
                   dist=norm1_similarity,
                   compare_comp=False,
                   weighted_avg :bool=False,
                   info_time:bool = False):
    
    starting_time = time.time() if info_time else None
    if n_component==0 and compare_comp:
        raise ValueError("Cannot compare PCA component for <=0 components")
    
    N,M = data.shape[:2]
    SPs_or = [[coor for coor in SP] for SP in SPs_or]

    pixelToSP = np.zeros((N,M), dtype=int)
    for k,SP in enumerate(SPs_or):
        for x,y in SP:
            pixelToSP[x,y] = k

    neighboors = [set() for _ in range(len(SPs_or))]
    borders = find_borders(SPs_or, (N,M), exterior=True)
    for k1 in range(len(borders)):
        for x,y in borders[k1]:
            k2 = pixelToSP[x,y]
            neighboors[k1].add(k2)

    return merge_SPs(SPs_or, neighboors, data, K,
               n_component=n_component, varFun=usedVarFun,
               compare_comp=compare_comp, dist=dist, weighted_avg=weighted_avg,
               starting_time=starting_time)
