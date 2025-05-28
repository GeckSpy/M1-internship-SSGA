# EntropyRateSupexpixel
# based on the work: https://github.com/mingyuliutw/EntropyRateSuperpixel

import numpy as np
import matplotlib.pyplot as plt
from classes import UnionFind, MinHeap
from typing import Callable


### Similarity functions
def basic_similarity(px:np.ndarray, py:np.ndarray, is_diagonal:bool)->float:
    """
    The original similarity function used
    - px, py define two spectral vector
    - is_diagonal: false if px and py are side by side
    """
    coef=np.sqrt(2) if is_diagonal else 1
    return coef * np.abs(np.average(px) - np.average(py))


def average_guassian_similarity(px: np.ndarray,
                                py: np.ndarray,
                                sigma: float,
                                is_diagonal: bool, 
                                fun: Callable[[np.ndarray, np.ndarray, bool], float]
                                )->float:
    twoSigmaSquared = 2*sigma*sigma
    return np.exp(-fun(px,py, is_diagonal)/twoSigmaSquared)


def complete_basic_similarity(px:np.ndarray, py:np.ndarray, is_diagonal:bool)->float:
    """
    The complete original similarity function used
    - px, py define two spectral vector
    - is_diagonal: false if px and py are side by side
    """
    return average_guassian_similarity(px,py,1, is_diagonal, basic_similarity)


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
Superpixel = list[list[tuple[int, int]]]
def find_superpixel(img: np.ndarray,
                    K: int,
                    lambda_coef: float,
                    similarity_function: Callable[[np.ndarray, np.ndarray, bool], float],
                    opt:bool=True
                    )->Superpixel:
    """
    - K: number of superpixel to find
    - lambda_coef (=0.5): balancing coefficient
    - opt (=true: 8-connected graph), (=false: 4-connected graph)
    - basic value: true
    """
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

    # Main loop
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
    return superpixels




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
            

def find_borders(L:Superpixel,
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
                            SP: Superpixel,
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


def plot_img_with_borders(img:np.ndarray, SP:Superpixel, color=[255,0,0,150]):
    """
    plot the given image with the superpixels' borders overlay
    """
    plt.imshow(img)
    plt.imshow(create_overlay_borders(img, SP, color=color))
    


def example():
    """
    Example of usage of entropy rate superpixel implementation
    """
    img = plt.imread("images/sun_umbrella.jpg")
    #plt.imshow([[np.average(y) for y in x] for x in img], cmap='gray')
    plt.imshow(img)
    plt.show()

    use_function = complete_basic_similarity
    res = find_superpixel(img, 100, 8*0.5, use_function, True)

    plot_img_with_borders(img, res, color = [255,0,0, 180])
    plt.show()

#example()