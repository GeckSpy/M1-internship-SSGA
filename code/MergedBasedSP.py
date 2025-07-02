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

    def f_inverse(y, x0=5.0):
        if y <= 0:
            raise ValueError("f(x) = x log(log(x)) is only defined for x > e")
        
        def equation(x):
            return f(x) - y
        
        result = root_scalar(equation, bracket=[np.e + 1e-5, 1e10], method='brentq')
        return result.root
    
    N,M = data.shape[0], data.shape[1]
    choosen_max = max(P_avg, f_inverse(n_component/gamma))
    return int(N*M/choosen_max)



def merge_SP(SPs:list[list[tuple[int,int]]],
             SPs_info:list[SPInfo],
             K:int,
             data:np.ndarray,
             simFun=comp_SP):
    
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


    n_component = SPs_info[0].n_component
    nb_cc = len(SPs)
    existing = [True for _ in range(nb_cc)]
    edges = [(u, v, simFun(SPs_info[u], SPs_info[v])) for u in range(nb_cc)
                for v in SPs_info[u].neighboor if u<v]
    edges.sort(key=lambda x:x[2])
    
    while nb_cc > K and len(edges)>0:
        k1,k2,_ = edges.pop(0)
        if existing[k1] and existing[k2]:
            existing[k2] = False
            SPs[k1] += SPs[k2]
            neighboors = SPs_info[k1].neighboor.union(SPs_info[k2].neighboor)
            SPs_info[k1] = SPInfo(SPs[k1], data, n_component)
            SPs_info[k1].neighboor = neighboors

            edges = [(u,v,w) for u,v,w in edges if u!=k1 and v!=k2 and u!=k2 and v!=k1 and existing[u] and existing[v]]
            for v in SPs_info[k1].neighboor:
                if v!=k1 and v!=k2 and existing[v]:
                    insert_sorted(edges, (k1,v,simFun(SPs_info[k1], SPs_info[v])))
            
            nb_cc -=1

    return [SP for i,SP in enumerate(SPs) if existing[i]], [SP_info for i,SP_info in enumerate(SPs_info) if existing[i]]



def compute_SP_by_merging(data, K, n_component=1, P_avg=20):
    N,M = data.shape[0], data.shape[1]
    K_or = computeKor(data, n_component=n_component, P_avg=P_avg)

    SPs = find_superpixel(data, K_or, lambda_coef="auto", simFun="norm1")
    pixelToSP = np.zeros((N,M), dtype=int)
    for k,SP in enumerate(SPs):
        for x,y in SP:
            pixelToSP[x,y] = k

    SPs_info = [SPInfo(SP, data, n_component) for SP in SPs]
    borders = find_borders(SPs, (N,M), exterior=True)
    for k1 in range(len(borders)):
        for x,y in borders[k1]:
            k2 = pixelToSP[x,y]
            SPs_info[k1].neighboor.add(k2)

    return merge_SP(SPs, SPs_info, K, data)
