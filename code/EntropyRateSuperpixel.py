# EntropyRateSupexpixel
# based on the work: https://github.com/mingyuliutw/EntropyRateSuperpixel

import time
import numpy as np
import matplotlib.pyplot as plt
from classes import UnionFind, MinHeap, Graph
from typing import Callable, Tuple


### Similarity functions
def basic_dist(is_diagonal:bool):
    return np.sqrt(2) if is_diagonal else 1

def average_similarity(px:np.ndarray, py:np.ndarray)->float:
    """
    The original similarity function used
    - px, py define two spectral vector
    """
    return np.abs(np.average(px) - np.average(py))


def norm2_similarity(px:np.ndarray, py:np.ndarray)->float:
    """
    The norm2 proposed similarity function
    - px, py define two spectral vector
    """
    return ((px-py)**2).sum()/len(px)


def norm1_similarity(px:np.ndarray, py:np.ndarray)->float:
    """
    The norm1 proposed similarity function
    - px, py define two spectral vector
    - is_diagonal: false if px and py are side by side
    """
    return np.abs(px-py).sum()/len(px)


def cosine_similarity(px:np.ndarray, py:np.ndarray,
                      x_norm=None, y_norm=None)->float:
    """
    Shifted cosine similarity
    - px, py define two spectral vector
    - is_diagonal: false if px and py are side by side
    """
    x_norm = np.linalg.norm(px) if type(x_norm)==type(None) else x_norm
    y_norm = np.linalg.norm(py) if type(y_norm)==type(None) else y_norm
    if x_norm==0 or y_norm==0:
        return -1
    cos_sim = np.dot(px,py)/(x_norm*y_norm)
    return (1-cos_sim)/2


def perason_correlation(px:np.ndarray, py:np.ndarray,
                        x_std=None, y_std=None,
                        x_norm=None, y_norm=None)->float:
    """
    Shifted Perason CorrelationAverageDist
    - px, py define two spectral vector
    - is_diagonal: false if px and py are side by side
    """
    sx = (px-np.average(px))/(np.std(px)+1e-8) if type(x_std)==type(None) else x_std
    sy = (py-np.average(py))/(np.std(py)+1e-8) if type(y_std)==type(None) else y_std
    return cosine_similarity(sx, sy, x_norm=x_norm, y_norm=y_norm)



def guassian(x, sigma=1):
    """
    compute guassian of x
    """
    twoSigmaSquared = 2*sigma*sigma
    return np.exp(-x/twoSigmaSquared)
    

SimFunType = Callable[[Tuple[int,int], Tuple[int,int], bool], float]
def create_CSF(simFun:str,
               data:np.ndarray
               )->SimFunType:
    """
    Create the Complete Similarity Function based of the name
    - data: the image to segment
    - simFun: name of wanted similarity function
        - average, norm1, norm2, cosine, perason
    """
    similarity_function = None
    usedData = data.copy()
    if simFun=="average":
        similarity_function = average_similarity
    elif simFun=="norm1":
        similarity_function = norm1_similarity
    elif simFun=="norm2":
        similarity_function = norm2_similarity
    if similarity_function!=None:
        return lambda x,y,b: guassian(basic_dist(b) * similarity_function(usedData[x], usedData[y]), 1)
    
    N,M = data.shape[0],data.shape[1]
    norms = np.zeros((N,M))
    if simFun=="cosine":
        for i in range(N):
            for j in range(M):
                norms[i,j] = np.linalg.norm(usedData[i,j])
        return lambda x,y,b: guassian(basic_dist(b) * 
                cosine_similarity(usedData[x], usedData[y], x_norm=norms[x], y_norm=[y]), 1)
    
    if simFun=="perason":
        for i in range(N):
            for j in range(M):
                usedData[i,j] = (usedData[i,j] - np.average(usedData[i,j]))/np.std(usedData[i,j])
                norms[i,j] = np.linalg.norm(usedData[i,j])
        return lambda x,y,b: guassian(basic_dist(b) * 
                                perason_correlation(usedData[x], usedData[y],
                                                    x_norm=norms[x], y_norm=norms[y],
                                                    x_std=usedData[x], y_std=usedData[y]), 1)

    raise ValueError("No valid similarity function name")
        




### Best models found for lambda coefficient
gamma = 0.15
def getLambdaAverage(K:int, N:int, M:int):
    """
    Return the lambda coefficient values of the best model for the basic similarity function (when data has been standardize)
    - K: number of wanted Superpixels
    - N,M: dimension of the image
    """
    return 0.114 * gamma* K * np.log(N*M*K)**1.158


def getLambdaNorm2(K:int, N:int, M:int):
    """
    Return the lambda coefficient values of the best model for the Norm 2 similarity function (when data has been standardize)
    - K: number of wanted Superpixels
    - N,M: dimension of the image
    """
    return 0.623 * gamma* K * np.log(N*M*K)**0.679


def getLambdaNorm1(K:int, N:int, M:int):
    """
    Return the lambda coefficient values of the best model for the Norm 1 similarity function (when data has been standardize)
    - K: number of wanted Superpixels
    - N,M: dimension of the image
    """
    return 0.221* gamma* K * np.log(N*M)**1.088


def getLambdaPerason(K:int, N:int, M:int):
    """
    Return the lambda coefficient values of the best model for the Perason similarity function (when data has been standardize)
    - K: number of wanted Superpixels
    - N,M: dimension of the image
    """
    return 0.595 * gamma * K * np.log(N*M*K)**0.293


nameToLambdaModel = {
    "average": getLambdaAverage,
    "norm1": getLambdaNorm1,
    "norm2": getLambdaNorm2,
    "perason": getLambdaPerason
}




### Build edges, loops and weight
class Edge:
    def __init__(self, u:int,v:int,w:float=0,gain:float=0):
        self.u:int = u
        self.v:int = v
        self.w:float = w          # Weight of the edge, i.e. the similarity between u and v
        self.gain:float = gain    # Gain of the edge, computed later
    
    def __str__(self)->str:
        return str((self.u, self.v, self.w, self.gain))
    
    def __ge__(self, other)->bool:
        """Not used but here to indicate that the priority of the edges are the gain"""
        if type(other)!=Edge:
            raise TypeError("unsupported operand between Edge and "+str(type(other)))
        return self.gain>other.gain

        
def build_edges(img: np.ndarray,
                opt: bool,
                similarity_function: SimFunType
                )->list[Edge]:
    """
    Compute all the edges and their similarity
    - opt=false -> 4-connected graph
    - opt=true -> 8-connected graph
    """
    n,m,_ = img.shape
    edges:list[Edge] = []
    for i in range(n):
        for j in range(m):
            u = i*m + j

            if j+1<m:
                v = i*m + j+1
                w = similarity_function((i,j), (i,j+1), False)
                edges.append(Edge(u,v,w))
            if i+1<n:
                v = (i+1)*m + j
                w = similarity_function((i,j), (i+1,j), False)
                edges.append(Edge(u,v,w))

                if opt:
                    if j+1<m:
                        v = (i+1)*m + j+1
                        w = similarity_function((i,j), (i+1,j+1), True)
                        edges.append(Edge(u,v,w))
                    if j-1>=0:
                        v = (i+1)*m + j-1
                        w = similarity_function((i,j), (i+1,j-1), True)
                        edges.append(Edge(u,v,w))
    return edges


def build_loops(img:np.ndarray, edges:list[Edge])->np.ndarray:
    """
    Build the weights of the loops
    """
    n,m,_ = img.shape
    loops = np.zeros(n*m)
    for e in edges:
        loops[e.u] += e.w
        loops[e.v] += e.w
    return loops



def build_loops_and_edges(img: np.ndarray,
                          opt: bool, 
                          similarity_function: SimFunType
                          )->tuple[np.ndarray, list[Edge], float]:
    """
    Build the edges, loops and their normalized weights
    - opt=false -> 4-connected graph
    - opt=true -> 8-connected graph
    """
    edges = build_edges(img, opt, similarity_function)
    loops:np.ndarray = build_loops(img, edges)
    total_weight = loops.sum()

    for e in edges:
        e.w /= total_weight

    return loops/total_weight, edges, total_weight






### entropy and balancing terms
def delta_entropy(a:float, b:float)->float:
    return -(a+b)*np.log(a+b) + a*np.log(a) + b*np.log(b)

def gain_entropy_rate(w:float, a:float, b:float)->float:
    if w+a<=0 or w+b<=0 or a<=0 or b<=0 or w<=0:
        return 0.0
    else:
        er = (w+a)*np.log(w+a) + (w+b)*np.log(w+b) -a*np.log(a) -b*np.log(b) -2*w*np.log(w)
        return er/np.log(2)

def gain_balancing(nb_vertices:int, a:int, b:int)->float:
    A,B = a/nb_vertices, b/nb_vertices
    return delta_entropy(A, B)/np.log(2) + 1





### Compute superpixel
def easyUpdateValueTree(heap: MinHeap,
                        uf: UnionFind,
                        edges: list[Edge],
                        loops:list[float],
                        nb_vertices:int,
                        balancing_term:float
                        )->bool:
    id = 0
    id_edge, old_gain = heap.array[id].get()
    e = edges[id_edge]
    ru, rv = uf.find(e.u), uf.find(e.v)
    if ru==rv:
        # will be deleted later
        heap.array[id].second = 0
    else:
        erGain = gain_entropy_rate(e.w, loops[e.u]-e.w, loops[e.v]-e.w)
        bGain = gain_balancing(nb_vertices, len(uf.component[ru]), len(uf.component[rv]))
        heap.array[id].second = erGain + balancing_term*bGain
        heap.array[id].second *= -1
    
    if(old_gain==heap.array[id].second):
        return False
    else:
        return True



def easyPartialUpdateTree(heap: MinHeap,
                          uf: UnionFind,
                          edges: list[Edge],
                          loops: np.ndarray,
                          nb_vertices:int,
                          balancing_term:float):
    """
    A special heap update structure that utilize the submodular property
    """
    while((not heap.isEmpty()) and easyUpdateValueTree(heap, uf, edges, loops, nb_vertices, balancing_term)):
        if(heap.array[0].second==0):
            # If the edge form a loop, remove it
            heap.pop()
        else:
            # Update the tree, 0 for the first (lowest priority) element
            heap.minHeapify(0)



### Lazy greedy
def find_superpixel(img: np.ndarray,
                    K,
                    lambda_coef = 0,
                    simFun:str = "average", 
                    updateLambda:bool = True,
                    custom_similarity_function: SimFunType=None,
                    diagonnalyConnected:bool=True,
                    shutSizeMatter:int =0,
                    newLambdaValue:int =0,
                    time_info:bool = False
                    )->list[list[tuple[int, int]]]:
    """
    - img: image to segment
    - K: number of superpixel to find. Can be a list to return multiple SPs selections
    - lambda_coef(>=0): balancing coefficient
        - 'auto' for automatic finding
    - simFun: name of wanted similarity function
        - average, norm1, norm2, cosine, perason
    - diagonnalyConnected (=true: 8-connected graph), (=false: 4-connected graph), default value: true
    - custom_similarity_function: a custom similarity function of type:
        - (int,int), (int,int), bool -> float
    """
    # Init var
    K_list = [K] if type(K)==int else [k for k in K]
    K_list.sort(key=lambda x:-x)
    if time_info:
        starting_time = time.time()
        dic_time = {}

    if simFun=="custom":
        CSF = custom_similarity_function
    else:
        CSF = create_CSF(simFun, img)


    N,M,_ = img.shape
    lambda_model = None
    if lambda_coef=="auto":
        if simFun not in nameToLambdaModel.keys():
            raise ValueError(simFun + "similarity function not supported for lambda 'auto'")
        else:
            lambda_model = nameToLambdaModel[simFun]
            lambda_coef = lambda_model(K_list[0], N,M)


    # Initialisation
    loops_weight, edges, _ = build_loops_and_edges(img, diagonnalyConnected, CSF)

    erGainArr = np.zeros(len(edges))
    bGainArr = np.zeros(len(edges))
    maxErGain, maxBGain = 0,0
    for i,e in enumerate(edges):
        erGainArr[i] = gain_entropy_rate(e.w, loops_weight[e.u], loops_weight[e.v])
        bGainArr[i] = gain_balancing(N*M, 1, 1)

        if erGainArr[i]>maxErGain:
            maxErGain = erGainArr[i]
        if bGainArr[i]>maxBGain:
            maxBGain = bGainArr[i]
    
    balancing = maxErGain/np.abs(maxBGain)
    lambda_balancing = lambda_coef * balancing
    for i,e in enumerate(edges):
        e.gain = erGainArr[i] + bGainArr[i]*lambda_balancing
    
    # Initialisation of Union-Find and MinHeap structure 
    uf = UnionFind(N*M)
    heap = MinHeap()
    for i,e in enumerate(edges):
        heap.insert(i, -e.gain)


    SPs_dic = {K:[] for K in K_list}
    # Main loop
    for K in K_list:
        if updateLambda and lambda_model!=None:
            lambda_coef = lambda_model(K,N,M)
            lambda_balancing = lambda_coef * balancing

        hasShutSize = False
        while uf.count>K and not heap.isEmpty():
            if (not hasShutSize) and uf.count<shutSizeMatter:
                hasShutSize = True
                if newLambdaValue<lambda_coef:
                    lambda_coef = newLambdaValue
                lambda_balancing = lambda_coef * balancing

            # Find the best edge to add
            best_edge = edges[heap.pop()[0]]
            # Add the edge to the graph
            ru,rv = uf.find(best_edge.u), uf.find(best_edge.v)
            if ru!=rv:
                uf.union(ru,rv)
                loops_weight[best_edge.u] -= best_edge.w
                loops_weight[best_edge.v] -= best_edge.w
                easyPartialUpdateTree(heap, uf, edges, loops_weight, N*M, lambda_balancing)
            
    
        # Compute pixels in each superpixels
        superpixel_component = [x for x in uf.component if len(x)>0]
        superpixels = [[] for _ in range(len(superpixel_component))]
        for i,superpixel in enumerate(superpixel_component):
            for j in superpixel:
                x = j//M
                y = j%M
                superpixels[i].append((x,y))
        SPs_dic[K] = superpixels
        if time_info:
            dic_time[K] = time.time() - starting_time

    res = SPs_dic[K_list[0]] if len(K_list)==1 else SPs_dic
    if time_info:
        return res, dic_time
    else:
        return res



### Compute borders of Superpixels
def find_border(l:list[tuple[int,int]],
                img_shape:tuple[int,int],
                exterior:bool =False
                )->list[tuple[int,int]]:
    """
    For l being a list of pixel's indices, return all the pixels at the borders of l
    - l: list of pixels
    - move_border: true for exterior border
    """
    n,m = img_shape
    mask = np.zeros(img_shape, dtype=int)
    for x,y in l:
        mask[x,y] = 255

    border = {
        (nx, ny) if exterior else (x,y)
        for x,y in l
        for dx,dy in [(-1,0), (1,0), (0,-1), (0,1)]
        if 0<=x+dx<n and 0<=y+dy<m and mask[x+dx, y+dy]==0
        for nx, ny in [(x+dx, y+dy)]
    }
    return list(border)
            

def find_borders(L:list[list[tuple[int, int]]],
                 img_shape:tuple[int,int],
                 exterior:bool=False
                 )->list[list[tuple[int,int]]]:
    """
    For L being a list of superpixel, return the list of borders of each superpixel
    - move_border: true for exterior border
    """
    borders = []
    for i,l in enumerate(L):
        borders.append(find_border(l, img_shape, exterior=exterior))
    return borders


def create_overlay_borders(img: np.ndarray,
                            SP: list[list[tuple[int, int]]],
                            color=[255,0,0,150],
                            exterior:bool = False):
    """
    Create an overaly image containing the borders of superpixels
    """
    if(len(color)==3):
        color = color + [1]

    if(len(img.shape))==2:
        n,m = img.shape
    else:
        n,m,_ = img.shape

    borders = find_borders(SP, (n,m), exterior=exterior)
    if len(color)==1:
        overlay = np.zeros((n,m), dtype=int)
    else:
        overlay = np.zeros((n,m, 4), dtype=int)
    for border in borders:
        for x,y in border:
            overlay[x,y] = color
    return overlay


def plot_img_with_borders(img:np.ndarray, SP:list[list[tuple[int, int]]], color=[255,0,0,150]):
    """
    plot the given image with the superpixels' borders overlay
    """
    plt.imshow(img)
    plt.imshow(create_overlay_borders(img, SP, color=color))
    


### Superpixel classes for data result
def groundtruthSegmentation(gt:np.ndarray):
    N,M = gt.shape
    graph = Graph(N*M)
    maxi = max(N,M)
    for i in range(N):
        for j in range(M):
            u = i*maxi + j

            if i+1<N and gt[i,j]==gt[i+1,j]:
                graph.add_edge(u, (i+1)*maxi + j)
            if j+1<M and gt[i,j]==gt[i,j+1]:
                graph.add_edge(u, i*maxi + j+1)

    cc = graph.composante_connexe()
    K = max(cc)+1
    SPs = [[] for _ in range(K)]
    img = np.zeros((N,M), dtype=int)

    for i in range(N):
        for j in range(M):
            u = i*maxi + j
            img[i,j] = cc[u]
            SPs[cc[u]].append((i,j))
    return SPs



class Superpixel:
    def __init__(self, liste, labels, gt, counting0=True):
        self.labels = [l for l in labels if not(not counting0 and l==0)]
        self.pixels = [coor for coor in liste if not(not counting0 and gt[coor]==0)]

        self.class_count = {l:0 for l in self.labels}
        for coor in self.pixels:
            self.class_count[gt[coor]] += 1

        self.guess = self.labels[0]
        for l in self.labels:
            if self.class_count[l] > self.class_count[self.guess]:
                self.guess = l

        self.proportion = self.class_count[self.guess]/len(self.pixels)
        self.isSingleClass = self.class_count[self.guess] == len(self.pixels)




class SuperpixelClassifier:
    def __init__(self, liste, labels, gt, counting0=True):
        self.counting0 = counting0
        self.labels = [l for l in labels if not(not counting0 and l==0)]

        self.liste = []
        self.pixels = []
        for l in liste:
            new_list = [coor for coor in l if not(not counting0 and gt[coor]==0)]
            if new_list!=[]:
                self.liste.append(new_list)
                self.pixels += new_list

        self.SPs:list[Superpixel] = [Superpixel(l, self.labels, gt, counting0) for l in self.liste]
        self.association:dict[tuple[int,int], Superpixel] = {}
        for i,SP in enumerate(self.SPs):
            for coor in SP.pixels:
                self.association[coor] = i

        self.data_class = {l:[] for l in labels}
        self.guess_map = np.zeros(gt.shape, dtype=int)
        for x,y in self.pixels:
            g = self.guess(x,y)
            self.guess_map[x,y] = g
            self.data_class[g].append((x,y))

    
    def getSP(self, x,y) -> Superpixel:
        return self.SPs[self.association[(x,y)]]
    

    def guess(self, x,y):
        return self.getSP(x,y).guess


    def predict(self, liste):
        return [self.guess(x,y) for x,y in liste]
    
    
    def accuracy(self, samples, labels):
        assert len(samples)==len(labels)
        prediction = self.predict(samples)
        return len([i for i in range(len(samples)) if prediction[i]==labels[i]])
    

    def singleClassCount(self):
        return len([i for i in range(len(self.SPs)) if self.SPs[i].isSingleClass])
    
    def singleClassProportion(self):
        return self.singleClassCount()/len(self.SPs)
    

    def averageProportion(self):
        return np.average([SP.proportion for SP in self.SPs])

    def averageWeightedProportion(self):
        sum = 0
        for SP in self.SPs:
            sum += len(SP.pixels) * SP.proportion
        return sum/len(self.pixels)
    
    def labelAccuracy(self, data_class, label):
        if label not in self.labels:
            return False
        goodGuessCount = 0
        for x,y in data_class[label][1]:
            if self.guess_map[x,y] == label:
                goodGuessCount += 1
        return goodGuessCount/len(data_class[label][1])
    
    def overallAccuracy(self, gt):
        goodGuessCount = 0
        for x,y in self.pixels:
            if gt[x,y] == self.guess_map[x,y]:
                goodGuessCount +=1
        return goodGuessCount/len(self.pixels)
    

    def averageAccuracy(self, gt):
        dic = {l:[0,0] for l in self.labels}
        for x,y in self.pixels:
            l = gt[x,y]
            dic[l][0] += 1
            if l==self.guess(x,y):
                dic[l][1] += 1
        
        return sum([e[1]/e[0] for e in dic.values()])/len(self.labels)
    


    def jaccard(self, gt, data_class, label, returnWeight=False):
        liste = data_class[label][1]
        if not self.counting0:
            liste = [coor for coor in liste if gt[coor]!=0]
            
        self_set = set(self.data_class[label])
        data_set = set(liste)
        inter = len(data_set.intersection(self_set))
        union = len(data_set.union(self_set))

        if returnWeight:
            return inter/union, len(liste)
        return inter/union
    
    
    def averageWeightedJaccard(self, gt, data_class):
        sum = 0
        for l in self.labels:
            jacc, weight = self.jaccard(gt, data_class, l, returnWeight=True)
            sum += weight*jacc
        return sum/(len(self.labels)*len(self.pixels))
    

    # ERS metrics
    def undersegmentationLabelError(self, label, data_class):
        liste = data_class[label][1]
        data_set = set(liste)
        sum = 0
        for l in self.labels:
            self_set = set(self.data_class[l])
            sum += len(self_set.intersection(data_set))
        return sum
    
    def undersegmentationError(self, data_class):
        sum = 0
        for l in self.labels:
            sum += self.undersegmentationLabelError(l, data_class)
        divisor = np.sum([len(item[1]) for key,item in data_class.keys() if key in self.labels])
        return sum/divisor
    

    def getSPs(self):
        return [SP.pixels for SP in self.SPs]

    def boundaryRecall(self, gt):
        N,M = gt.shape
        gtSPs = groundtruthSegmentation(gt)
        selfSPs = self.getSPs()

        gtBoundaries = []
        for SP in gtSPs:
            gtBoundaries += find_border(SP, gt.shape)

        selfBoundaries = create_overlay_borders(gt, selfSPs, color=[1])
        sum = 0
        neig = [(-1,-1),(-1,0),(-1,1),(0,-1), (0,0), (0, 1),(1,-1), (1,0), (1,1)]
        for x,y in gtBoundaries:
            b = False
            for dx,dy in neig:
                if 0<=x+dx<N and 0<=y+dy<M and selfBoundaries[x+dx,y+dy]==1:
                    b=True
                    break
            if b:
                sum+=1
        return sum/len(gtBoundaries)
        
    





### Example of usage
def example1():
    """
    Example of usage of entropy rate superpixel implementation
    """
    img = plt.imread("images/sun_umbrella.jpg")
    #plt.imshow([[np.average(y) for y in x] for x in img], cmap='gray')
    #plt.imshow(img)
    #plt.show()

    res = find_superpixel(img, 100,lambda_coef=4, simFun="average")
    l = [len(l) for l in res]
    print(l)
    print(np.sum(l))

    plot_img_with_borders(img, res, color = [255,0,0, 180])
    plt.show()


def example2():
    """
    Example of usage of entropy rate superpixel implementation with multiple Ks
    """
    img = plt.imread("images/low_flower.png")

    Ks = [10, 20, 40, 30]
    res = find_superpixel(img, Ks, lambda_coef=4, simFun="norm2", diagonnalyConnected=False)
    Ks.sort(key=lambda x:-x)

    fig, axs = plt.subplots(1, len(Ks), figsize=(20,10))
    for i,K in enumerate(Ks):
        axs[i].imshow(img)
        mask = create_overlay_borders(img, res[i])
        axs[i].imshow(mask) 
        axs[i].axis("off")
        axs[i].title.set_text("K = "+str(K))
    plt.show()

#example1()
#example2()