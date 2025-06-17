# EntropyRateSupexpixel
# based on the work: https://github.com/mingyuliutw/EntropyRateSuperpixel

import numpy as np
import matplotlib.pyplot as plt
from classes import UnionFind, MinHeap
from typing import Callable


### Similarity functions
def basic_dist(is_diagonal:bool):
    return 1/np.sqrt(2) if is_diagonal else 1

def basic_similarity(px:np.ndarray, py:np.ndarray, is_diagonal:bool)->float:
    """
    The original similarity function used
    - px, py define two spectral vector
    - is_diagonal: false if px and py are side by side
    """
    return basic_dist(is_diagonal) * np.abs(np.average(px) - np.average(py))


def norm2_similarity(px:np.ndarray, py:np.ndarray, is_diagonal:bool)->float:
    """
    The norm2 proposed similarity function
    - px, py define two spectral vector
    - is_diagonal: false if px and py are side by side
    """
    return basic_dist(is_diagonal) * ((px-py)**2).sum()/len(px)


def norm1_similarity(px:np.ndarray, py:np.ndarray, is_diagonal:bool)->float:
    """
    The norm1 proposed similarity function
    - px, py define two spectral vector
    - is_diagonal: false if px and py are side by side
    """
    return basic_dist(is_diagonal) * np.abs(px-py).sum()/len(px)


def cosine_similarity(px:np.ndarray, py:np.ndarray, is_diagonal:bool)->float:
    """
    """
    return (1-np.dot(px,py)/(np.linalg.norm(px)*np.linalg.norm(py)))/2



def guassian_similarity(
        px: np.ndarray,
        py: np.ndarray,
        sigma: float,
        is_diagonal: bool, 
        fun: Callable[[np.ndarray, np.ndarray, bool], float]
        )->float:
    """
    Return exp(-fun(px, py, is_diagoanl)/2sigma²)
    """
    twoSigmaSquared = 2*sigma*sigma
    return np.exp(-fun(px,py, is_diagonal)/twoSigmaSquared)


def complete_basic_similarity(px:np.ndarray, py:np.ndarray, is_diagonal:bool)->float:
    """
    The complete original similarity function used
    - px, py define two spectral vector
    - is_diagonal: false if px and py are side by side
    """
    return guassian_similarity(px,py,1, is_diagonal, basic_similarity)


def complete_norm2_similarity(px:np.ndarray, py:np.ndarray, is_diagonal:bool)->float:
    return guassian_similarity(px,py,1, is_diagonal, norm2_similarity)

def complete_norm1_similarity(px:np.ndarray, py:np.ndarray, is_diagonal:bool)->float:
    return guassian_similarity(px,py,1, is_diagonal, norm1_similarity)


def complete_cosine_similarity(px:np.ndarray, py:np.ndarray, is_diagonal:bool)->float:
    return guassian_similarity(px,py,1, is_diagonal, cosine_similarity)



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
                similarity_function: Callable[[np.ndarray, np.ndarray, bool], float]
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
                w = similarity_function(img[i,j], img[i,j+1], False)
                edges.append(Edge(u,v,w))
            if i+1<n:
                v = (i+1)*m + j
                w = similarity_function(img[i,j], img[i+1,j], False)
                edges.append(Edge(u,v,w))

                if opt:
                    if j+1<m:
                        v = (i+1)*m + j+1
                        w = similarity_function(img[i,j], img[i+1,j+1], True)
                        edges.append(Edge(u,v,w))
                    if j-1>=0:
                        v = (i+1)*m + j-1
                        w = similarity_function(img[i,j], img[i+1,j-1], True)
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
                          similarity_function: Callable[[np.ndarray, np.ndarray, bool], float]
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
                    lambda_coef: float,
                    similarity_function: Callable[[np.ndarray, np.ndarray, bool], float],
                    opt:bool=True
                    )->list[list[tuple[int, int]]]:
    """
    - K: number of superpixel to find. Can be a list to return multiple SPs selections
    - lambda_coef (=0.5): balancing coefficient
    - opt (=true: 8-connected graph), (=false: 4-connected graph), default value: true
    """
    if type(K)==int:
        K_list = [K]
    else:
        K_list = [k for k in K]
    K_list.sort(key=lambda x:-x)

    # Initialisation
    n,m,_ = img.shape
    loops_weight, edges, _ = build_loops_and_edges(img, opt, similarity_function)

    erGainArr = np.zeros(len(edges))
    bGainArr = np.zeros(len(edges))
    maxErGain, maxBGain = 0,0
    for i,e in enumerate(edges):
        erGainArr[i] = gain_entropy_rate(e.w, loops_weight[e.u], loops_weight[e.v])
        bGainArr[i] = gain_balancing(n*m, 1, 1)

        if erGainArr[i]>maxErGain:
            maxErGain = erGainArr[i]
        if bGainArr[i]>maxBGain:
            maxBGain = bGainArr[i]
    

    balancing = lambda_coef*maxErGain/np.abs(maxBGain)
    for i,e in enumerate(edges):
        e.gain = erGainArr[i] + bGainArr[i]*balancing
    
    # Initialisation of Union-Find and MinHeap structure 
    uf = UnionFind(n*m)
    heap = MinHeap()
    for i,e in enumerate(edges):
        heap.insert(i, -e.gain)


    SPs_list = []
    # Main loop
    for K in K_list:
        while uf.count>K and not heap.isEmpty():
            # Find the best edge to add
            best_edge = edges[heap.pop()[0]]
            # Add the edge to the graph
            ru,rv = uf.find(best_edge.u), uf.find(best_edge.v)
            if ru!=rv:
                uf.union(ru,rv)
                loops_weight[best_edge.u] -= best_edge.w
                loops_weight[best_edge.v] -= best_edge.w
                easyPartialUpdateTree(heap, uf, edges, loops_weight, n*m, balancing)
            
    
        # Compute pixels in each superpixels
        superpixel_component = [x for x in uf.component if len(x)>0]
        superpixels = [[] for _ in range(len(superpixel_component))]
        for i,superpixel in enumerate(superpixel_component):
            for j in superpixel:
                x = j//m
                y = j%m
                superpixels[i].append((x,y))
        SPs_list.append(superpixels)

    if len(K_list)==1:
        return SPs_list[0]
    else:
        return SPs_list



### Compute borders of Superpixels
def find_border(l:list[tuple[int,int]],
                img_shape:tuple[int,int]
                )->list[tuple[int,int]]:
    """
    For l being a list of pixel's indices, return all the pixels at the borders of l
    """
    n,m = img_shape
    border = []
    mask = np.zeros(img_shape, dtype=int)
    for x,y in l:
        mask[x,y] = 255

    for x,y in l:
        if x>0:
            if mask[x-1][y]==0:
                border.append((x,y))
        if x+1<n:
            if mask[x+1][y]==0:
                border.append((x,y))

        if y>0:
            if mask[x][y-1]==0:
                border.append((x,y))
        if y+1<m:
            if mask[x][y+1]==0:
                border.append((x,y))
    return border
            

def find_borders(L:list[list[tuple[int, int]]],
                 img_shape:tuple[int,int]
                 )->list[list[tuple[int,int]]]:
    """
    For L being a list of superpixel, return the list of borders of each superpixel
    """
    borders = []
    for i,l in enumerate(L):
        borders.append(find_border(l, img_shape))
    return borders


def create_overlay_borders(img: np.ndarray,
                            SP: list[list[tuple[int, int]]],
                            color=[255,0,0,150]):
    """
    Create an overaly image containing the borders of superpixels
    """
    if(len(color)==3):
        color = color + [1]

    if(len(img.shape))==2:
        n,m = img.shape
    else:
        n,m,_ = img.shape

    borders = find_borders(SP, (n,m))
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
        return sum/len(self.labels)




### Example of usage
def example1():
    """
    Example of usage of entropy rate superpixel implementation
    """
    img = plt.imread("images/sun_umbrella.jpg")
    #plt.imshow([[np.average(y) for y in x] for x in img], cmap='gray')
    #plt.imshow(img)
    #plt.show()

    use_function = complete_basic_similarity
    res = find_superpixel(img, 100, 8*0.5, use_function, True)
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

    use_function = complete_basic_similarity
    Ks = [10, 20, 40, 30]
    res = find_superpixel(img, Ks, 8*0.5, use_function, True)
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