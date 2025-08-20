import numpy as np

# Classes
class Pair:
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def __str__(self):
        return "(" + str(self.first) + ", " + str(self.second) +")"
    
    def get(self):
        return self.first, self.second

class MinHeap:
    def __init__(self):
        self.array:list[Pair] = []

    def __str__(self):
        return str(self.array)

    def left(self, i):
        return 2*i+1
    
    def right(self, i):
        return 2*i+2

    def isEmpty(self):
        return self.array==[]

    def insert(self, elt, val):
        """Insert a new element into the Min Heap."""
        self.array.append(Pair(elt, val))
        i = len(self.array) - 1
        while i>0 and self.array[(i - 1) // 2].second > self.array[i].second:
            self.array[i], self.array[(i - 1) // 2] = self.array[(i - 1) // 2], self.array[i]
            i = (i - 1) // 2


    def delete(self, i):
        "delete element at index i"
        if i == -1:
            return False
        self.array[i] = self.array[-1]
        self.array.pop()
        #self.minHeapify(i)
        while True:
            left = self.left(i)
            right = self.right(i)
            smallest = i
            if left<len(self.array) and self.array[left].second < self.array[smallest].second:
                smallest = left
            if right<len(self.array) and self.array[right].second < self.array[smallest].second:
                smallest = right
            if smallest != i:
                self.array[i], self.array[smallest] = self.array[smallest], self.array[i]
                i = smallest
            else:
                break
        return True
    

    def delete_element(self, elt):
        """Delete a specific element from the Min Heap."""
        i = -1
        for j in range(len(self.array)):
            if self.array[j].first == elt:
                i = j
                break
        return self.delete(i)

    
    def minHeapify(self, i):
        """Heapify function to maintain the heap property.""" 
        n = len(self.array)
        smallest = i
        left = self.left(i)
        right = self.right(i)

        if left<n and self.array[left].second < self.array[smallest].second:
            smallest = left
        if right<n and self.array[right].second < self.array[smallest].second:
            smallest = right
        if smallest != i:
            self.array[i], self.array[smallest] = self.array[smallest], self.array[i]
            self.minHeapify(smallest)


    def getMin(self):
        return self.array[0].get() if self.array!=[] else None

    def pop(self):
        if self.array!=[]:
            pair = self.array[0].get()
            self.delete(0)
            return pair
        else:
            return None
    
    def fancy_print(self):
        def aux(i, nb_space):
            if i<len(self.array):
                aux(self.left(i), nb_space+1)
                print(nb_space*"  ", self.array[i])
                aux(self.right(i), nb_space+1)
        aux(0, 0)


# Normalizing HSI dataset
def normalized_data(data:np.ndarray) -> np.ndarray:
    """
    Normalize the data
    """
    N,M,B = data.shape
    res = np.zeros(data.shape)
    for b in range(B):
        img = data[:,:,b]
        res[:,:,b] = (img - img.min())/(img.max()-img.min())
    return res



# Similarity functions
def norm1_similarity(px:np.ndarray, py:np.ndarray)->float:
    """
    The norm1 proposed similarity function
    - px, py define two spectral vector
    - is_diagonal: false if px and py are side by side
    """
    return np.abs(px-py).sum()/len(px)

def norm2_similarity(px:np.ndarray, py:np.ndarray)->float:
    """
    The norm2 proposed similarity function
    - px, py define two spectral vector
    """
    return ((px-py)**2).sum()/len(px)

### Std-based similarity function
def normalize(vec):
    arr = np.array(vec)
    mini = arr.min()
    maxi = arr.max()
    if mini==maxi:
        return arr/len(arr)
    return (arr-mini)/(maxi-mini)

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

def stdFtestnorm1(clusters:list, dtc=None, averages=None):
    # High = well separated
    coeff = 2
    K = len(clusters)
    sizes = [len(cluster) for cluster in clusters]
    n = np.sum(sizes)
    if type(averages)==type(None):
        averages = compute_averages(clusters, dtc=dtc)

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



# SNIC-based SPS for HSI dataset
def find_seeds(height:int, width:int, K:int)->tuple[int, list[int], list[int]]:
    """
    Return the seeds using hexagonal image splitting
    """
    hex_area = (height * width) / K
    r = np.sqrt((2 * hex_area) / (3 * np.sqrt(3)))

    dx = 3 * r / 2
    dy = np.sqrt(3) * r
    kx, ky = [], []

    y = r
    row = 0
    while y < height:
        offset_x = r + (row % 2) * (3 * r / 4)
        x = offset_x
        while x < width:
            kx.append(int(x))
            ky.append(int(y))
            x += dx
        y += dy
        row += 1

    return len(kx), kx, ky

def mySNIC(data:np.ndarray, K:int)->list[list[tuple[int,int]]]:
    """
    The proposed SNIC improvment algorithm for more images type, designed for hyperspectral images
    """
    height, width, B = data.shape
    dx8 = [-1, 0, 1, 0, -1, 1, 1, -1]
    dy8 = [0, -1, 0, 1, -1, -1, 1, 1]
    
    numk, cx, cy = find_seeds(width, height, K)
    labels = -1 * np.ones((height, width), dtype=int)

    ks = np.zeros((numk, B))
    kx = np.zeros(numk)
    ky = np.zeros(numk)
    ksize = np.zeros(numk)

    # Priority queue
    heap = MinHeap()
    for k in range(numk):
        heap.insert((cx[k], cy[k], k), 0)

    CONNECTIVITY = 4
    #CONNECTIVITY = 8

    while not heap.isEmpty():
        (x,y,k),_ = heap.pop()

        if labels[x,y] < 0:
            labels[x,y] = k

            for b in range(B):
                ks[k][b] += data[x,y,b]

            kx[k] += x
            ky[k] += y
            ksize[k] += 1.0

            for p in range(CONNECTIVITY):
                xx = x + dx8[p]
                yy = y + dy8[p]
                if 0 <= xx < height and 0 <= yy < width:
                    if labels[xx,yy]<0:
                        means = np.array([ks[k][b]/ksize[k] for b in range(B)])
                        colordist = norm2_similarity(means, data[xx,yy,:])
                        slicdist = colordist
                        heap.insert((xx,yy,k), slicdist)

    SPs = [[] for _ in range(numk)]
    for i in range(height):
        for j in range(width):
            SPs[labels[i,j]].append((i,j))
    return SPs





# Merge-Based information
def computeKor(N:int, M:int, P_avg:float=25)->int:
    return int(N*M/P_avg)

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


def computeMergeBasedInfo(data:np.ndarray):
    N,M = data.shape[0], data.shape[1]
    K_or = computeKor(N,M)

    SPs_or = mySNIC(data, K_or)

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



# Merge-Based segmentation
def center_distances(SP):
    def dist_squared(c1, c2):
        return (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2

    center = np.average(SP, axis=0)
    return np.array([dist_squared(coor, center) for coor in SP])


def merge_SPs(SPs_or :list[list[tuple[int,int]]],
              neighboors_or :list[set[int]],
              trainData :np.ndarray,
              K :int):
    
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


    SPs = [SP.copy() for SP in SPs_or]
    neighboors:list[set] = [neighboor.copy() for neighboor in neighboors_or]
    TSs = [[trainData[coor] for coor in SP] for SP in SPs]
    averages = compute_averages(TSs, dtc=None)

    def simFun(k1, k2):
        dtc = [center_distances(SP) for SP in [SPs[k1], SPs[k2]]]
        return stdFtestnorm1([TSs[k1],TSs[k2]], dtc=dtc, averages=[averages[k1],averages[k2]])
        
            
        
    

    nb_cc = len(SPs)
    existing = [True for _ in range(nb_cc)]
    edges = [(u, v, simFun(u, v)) for u in range(nb_cc)
                for v in neighboors[u] if u<v]
    edges.sort(key=lambda x:x[2])
    
    SPsDic = {K:[] for K in Ks}
    for K in Ks:
        while nb_cc > K and len(edges)>0:
            k1,k2,_ = edges[0]
            if existing[k1] and existing[k2]:
                existing[k2] = False
                SPs[k1] += SPs[k2]
                TSs[k1] += TSs[k2]
                averages[k1] = np.average(TSs[k1], axis=0)

                SPs[k2] = None
                for k in neighboors[k2]:
                    if existing[k]:
                        neighboors[k] = neighboors[k] - set([k2]) 
                        neighboors[k].add(k1)
                neighboors[k1] = set([x for x in neighboors[k1].union(neighboors[k2]) if x!=k1 and x!=k2 and existing[x]])
                neighboors[k2] = None


                new_edges = []
                for u,v,w in edges:
                    if u==k1 or u==k2 or v==k1 or v==k2:
                        pass
                    elif existing[u] and existing[v]:
                        new_edges.append((u,v,w))
                edges = new_edges


                for v in neighboors[k1]:
                    sf = simFun(k1,v)
                    insert_sorted(edges, (k1,v, sf))


                nb_cc -=1

            else:
                print("ERROR")
                assert False

        SPsDic[K] = [[coor for coor in SP] for i,SP in enumerate(SPs) if existing[i]]

    return SPsDic[Ks[0]] if len(Ks)==1 else SPsDic
    


def SMCSF(data, K):
    trainData = normalized_data(data)
    infos = computeMergeBasedInfo(trainData)
    SPs_or, neighboors = infos

    return merge_SPs(SPs_or, neighboors, trainData, K)
