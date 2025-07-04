import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import root_scalar

from EntropyRateSuperpixel import norm1_similarity, find_superpixel, find_borders



### Superpixel Characterization
class SPInfo:
    def __init__(self, pixels:list[tuple[int,int]], data:np.ndarray, n_component:int):
        self.time_series = np.array([data[x,y] for x,y in pixels])
        #self.mean = np.average(self.time_series, axis=0)
        #self.std = np.std(self.time_series, axis=0)
        self.neighboor = set()

        self.n_component = min(n_component, min(self.time_series.shape))

        if len(pixels)==1:
            self.components = self.time_series
        else:
            self.pca = PCA(n_components=self.n_component)
            self.coeffs = self.pca.fit_transform(self.time_series)
            self.components = np.array([self.pca.components_[i]+self.pca.mean_ for i in range(self.n_component)])     



### Superpixels similarity function
def sim_comp(comp1:np.ndarray, comp2:np.ndarray, simFun=norm1_similarity)->float:
    n,_ = comp1.shape
    m,_ = comp2.shape
    distances = 0
    for i in range(n):
        for j in range(m):
            distances += simFun(comp1[i], comp2[j])
    return distances/(n*m)


def comp_SP(info1:SPInfo, info2:SPInfo, simFun=norm1_similarity)->float:
    compSim = sim_comp(info1.components, info2.components, simFun=simFun)
    return compSim


def compute_medoid(group:list, simFun=norm1_similarity):
    n,_ = group.shape
    distances = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i!=j:
                distances[i] += simFun(group[i], group[j])
    metroid_index = np.argmin(distances)
    return group[metroid_index]


### Algorithm
# Computing Kor
def computeKor(data:np.ndarray, P_avg:float=20, n_component:int=1, gamma:float=0.15)->int:
    def f(x):
        return x * np.log(np.log(x))

    def f_inverse(y):
        if y <= 0:
            raise ValueError("f(x) = x log(log(x)) is only defined for x > e")
        
        def equation(x):
            return f(x) - y
        
        result = root_scalar(equation, bracket=[np.e + 1e-5, 1e10], method='brentq')
        return result.root
    
    N,M = data.shape[0], data.shape[1]
    choosen_max = max(P_avg, f_inverse(n_component/gamma))
    return int(N*M/choosen_max)




def merge_SPs(SPs_or, neighboors_or, K, trainData, n_component=0, varFun=anovaFtest):
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


    def simFun(group1, group2, n_component=n_component):
        if n_component==0:
            return varFun([group1, group2])
        else:
            TS = np.array([trainData[coor] for group in [group1, group2] for coor in group])
            n_component = min(n_component, min(TS.shape))
            pca = PCA(n_components=n_component)
            coeffs = pca.fit_transform(TS)

            clusters = [[], []]
            for i in range(len(group1)):
                clusters[0].append(coeffs[i])
            for i in range(len(group2)):
                clusters[1].append(coeffs[i+len(group1)])
            #clusters = [np.array(cluster) for cluster in clusters]
            return varFun(clusters)


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



def mergedBasedSegmentation(data, K, n_component=0, usedVarFun=anovaFtest):
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

    return merge_SPs(SPs_or, neighboors, K, data,
               n_component=n_component, varFun=usedVarFun)
