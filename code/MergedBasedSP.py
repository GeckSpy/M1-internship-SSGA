import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import root_scalar

from EntropyRateSuperpixel import norm1_similarity, find_superpixel, find_borders
from MultilevelSP import anovaFtest

### Superpixels similarity function
def AverageDist(components, dist=norm1_similarity)->float:
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




def compute_medoid(group:list, dist=norm1_similarity):
    n,_ = group.shape
    distances = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i!=j:
                distances[i] += dist(group[i], group[j])
    metroid_index = np.argmin(distances)
    return group[metroid_index]


### Algorithm
# Computing Kor
def computeKor(N,M, P_avg:float=35, n_component:int=0, gamma:float=0.15)->int:
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




def merge_SPs(SPs_or, neighboors_or, K, trainData, n_component=0, varFun=anovaFtest, compare_comp=False, dist=norm1_similarity):
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


    def simFun(group1, group2, n_component=n_component, compare_comp=compare_comp):
        if n_component==0:
            return varFun([group1, group2], dist=dist)
        if compare_comp:
            clusters = []
            for group in [group1, group2]:
                TS = np.array([trainData[coor] for coor in group])
                n_component = min(n_component, min(TS.shape))
                pca = PCA(n_components=n_component)
                pca.fit_transform(TS)
                clusters.append(pca.components_ + pca.mean_)
            return varFun(clusters, dist=dist)
            
        TS = np.array([trainData[coor] for group in [group1, group2] for coor in group])
        n_component = min(n_component, min(TS.shape))
        pca = PCA(n_components=n_component)
        coeffs = pca.fit_transform(TS)

        clusters = [[], []]
        for i in range(len(group1)):
            clusters[0].append(coeffs[i])
        for i in range(len(group2)):
            clusters[1].append(coeffs[i+len(group1)])
        clusters = [np.array(cluster) for cluster in clusters]
        return varFun(clusters, dist=dist)


    SPs = [SP.copy() for SP in SPs_or]
    neighboors:list[set] = [neighboor.copy() for neighboor in neighboors_or]

    nb_cc = len(SPs)
    existing = [True for _ in range(nb_cc)]
    edges = [(u, v, simFun(SPs[u], SPs[v])) for u in range(nb_cc)
                for v in neighboors[u] if u<v]
    edges.sort(key=lambda x:x[2])
    
    while nb_cc > K and len(edges)>0:
        k1,k2,_ = edges.pop(0)
        if existing[k1] and existing[k2]:
            existing[k2] = False
            SPs[k1] += SPs[k2]
            SPs[k2] = None
            neighboors[k1] = neighboors[k1].union(neighboors[k2])
            neighboors[k2] = None

            edges = [(u,v,w) for u,v,w in edges if u!=k1 and v!=k2 and u!=k2 and v!=k1 and existing[u] and existing[v]]
            for v in neighboors[k1]:
                if v!=k1 and existing[v]:
                    insert_sorted(edges, (k1,v, simFun(SPs[k1], SPs[v])))
            
            nb_cc -=1

    return [SP for i,SP in enumerate(SPs) if existing[i]]



def computeMergeBasedInfo(data, n_component=0):
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


def mergedBasedSegmentation(data, K, n_component=0, usedVarFun=anovaFtest, infos=None, compare_comp=False):
    if n_component==0 and compare_comp:
        raise ValueError("Cannot compare PCA component for <=0 components")
    if infos==None:
        infos = computeMergeBasedInfo(data, n_component=n_component)
    SPs_or, neighboors = infos

    return merge_SPs(SPs_or, neighboors, K, data,
               n_component=n_component, varFun=usedVarFun, compare_comp=compare_comp)
