import numpy as np
import heapq
from skimage import color

from skimage.segmentation import slic
from skimage.util import img_as_float
from EntropyRateSuperpixel import norm1_similarity, norm2_similarity
from classes import MinHeap


# SLIC algorithm
def SLIC(data:np.ndarray, K:int, compactness:float=10)->list[list[tuple[int,int]]]:
    # The classic SLIC algorithm with the used Supepixel Segmentation format
    n,m = data.shape[:2]
    segments = slic(img_as_float(data), n_segments=K, compactness=compactness, start_label=0)
    SPs = [[] for _ in range(np.max(segments)+1)]
    for i in range(n):
        for j in range(m):
            SPs[segments[i,j]].append((i,j))
    return SPs


# SNIC algorithm
def rgb_to_lab_image(img:np.ndarray):
    """
    Converts an RGB image to LAB using skimage.
    Expects image as numpy array with shape (H, W, 3), dtype uint8.
    """
    img_float = img.astype(np.float32) / 255.0
    lab = color.rgb2lab(img_float)
    return lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]  # L, A, B


### Seeds functions
def find_square_seeds(width:int, height:int, numk:int)->tuple[int, list[int], list[int]]:
    """
    Return the seeds with the classic SNIC seeds algorithm
    """
    sz = width * height
    gridstep = int(np.sqrt(sz / numk) + 0.5)
    halfstep = gridstep // 2

    xsteps = width // gridstep
    ysteps = height // gridstep
    err1 = abs(xsteps * ysteps - numk)
    err2 = abs((width // (gridstep - 1)) * (height // (gridstep - 1)) - numk)
    
    if err2 < err1:
        gridstep -= 1
        xsteps = width // gridstep
        ysteps = height // gridstep

    numk = xsteps * ysteps
    kx, ky = [], []
    for y in range(halfstep, height, gridstep):
        for x in range(halfstep, width, gridstep):
            if y <= height - halfstep and x <= width - halfstep:
                kx.append(x)
                ky.append(y)
    return numk, kx, ky


def find_hexagonal_seeds(height:int, width:int, K:int)->tuple[int, list[int], list[int]]:
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


def find_seeds(N:int, M:int, K:int, shape="square")->tuple[int, list[int], list[int]]:
    if shape=="square":
        return find_square_seeds(N,M,K)
    elif shape=="hexagon":
        return find_hexagonal_seeds(N,M,K)
    else:
        raise ValueError("attribute shape must be either 'square' or 'hexagon'")


### main function
def run_snic(lv:np.ndarray,
             av:np.ndarray,
             bv:np.ndarray,
             width:int, height:int, innumk:int,
             compactness:float,
             shape:str="square"
             )->tuple[np.ndarray, int]:
    """
    Run the classic SNIC algorithm
    """
    sz = width * height
    dx8 = [-1, 0, 1, 0, -1, 1, 1, -1]
    dy8 = [0, -1, 0, 1, -1, -1, 1, 1]
    dn8 = [-1, -width, 1, width, -1 - width, 1 - width, 1 + width, -1 + width]
    
    numk, cx, cy = find_seeds(width, height, innumk, shape=shape)
    labels = -1 * np.ones(sz, dtype=np.int32)
    kl = np.zeros(numk)
    ka = np.zeros(numk)
    kb = np.zeros(numk)
    kx = np.zeros(numk)
    ky = np.zeros(numk)
    ksize = np.zeros(numk)

    # Priority queue
    pq = []
    for k in range(numk):
        i = (cx[k] << 16) | cy[k]
        heapq.heappush(pq, (0, i, k))  # (distance, index, label)

    CONNECTIVITY = 4
    #CONNECTIVITY = 8
    M = compactness
    invwt = (M * M * numk) / float(sz)

    while pq:
        d, packed_i, k = heapq.heappop(pq)
        x = (packed_i >> 16) & 0xffff
        y = packed_i & 0xffff
        i = y * width + x

        if labels[i] < 0:
            labels[i] = k
            kl[k] += lv[i]
            ka[k] += av[i]
            kb[k] += bv[i]
            kx[k] += x
            ky[k] += y
            ksize[k] += 1.0

            for p in range(CONNECTIVITY):
                xx = x + dx8[p]
                yy = y + dy8[p]
                if 0 <= xx < width and 0 <= yy < height:
                    ii = i + dn8[p]
                    if 0 <= ii < sz and labels[ii] < 0:
                        lmean = kl[k] / ksize[k]
                        amean = ka[k] / ksize[k]
                        bmean = kb[k] / ksize[k]
                        xmean = kx[k] / ksize[k]
                        ymean = ky[k] / ksize[k]

                        ldiff = lmean - lv[ii]
                        adiff = amean - av[ii]
                        bdiff = bmean - bv[ii]
                        xdiff = xmean - xx
                        ydiff = ymean - yy

                        colordist = ldiff**2 + adiff**2 + bdiff**2
                        spatialdist = xdiff**2 + ydiff**2
                        slicdist = colordist + invwt * spatialdist


                        packed_ii = (xx << 16) | yy
                        heapq.heappush(pq, (slicdist, packed_ii, k))

    # Fill in any unlabelled pixels
    if labels[0] < 0:
        labels[0] = 0
    for y in range(1, height):
        for x in range(1, width):
            i = y * width + x
            if labels[i] < 0:
                if labels[i - 1] >= 0:
                    labels[i] = labels[i - 1]
                elif labels[i - width] >= 0:
                    labels[i] = labels[i - width]

    return labels.reshape((height, width)), numk



def SNIC(image:np.ndarray, num_superpixels:int=100, compactness:float=10.0, shape="square"):
    """
    Main interface for SNIC segmentation.
    
    Parameters:
    - image: RGB image as numpy array (H, W, 3), dtype=uint8
    - num_superpixels: desired number of superpixels
    - compactness: tradeoff parameter between color and spatial proximity
    
    Returns:
    - labels: label map of shape (H, W)
    - num_actual_superpixels: number of seeds used
    """
    height, width = image.shape[:2]
    lvec, avec, bvec = rgb_to_lab_image(image)
    lv = lvec.flatten()
    av = avec.flatten()
    bv = bvec.flatten()
    
    labels, numk = run_snic(lv, av, bv, width, height, num_superpixels, compactness, shape=shape)

    SPs = [[] for _ in range(numk)]
    for i in range(height):
        for j in range(width):
            SPs[labels[i,j]].append((i,j))

    return SPs


# My SNIC
### classic SNIC algorithm (made for RBG images)
def myRBGSNIC(data:np.ndarray, K:int, compactness:int=0, shape="square")->list[list[tuple[int,int]]]:
    """
    The classic SNIC algorithm using more understandable code
    """
    height, width = data.shape[:2]
    sz = width * height
    dx8 = [-1, 0, 1, 0, -1, 1, 1, -1]
    dy8 = [0, -1, 0, 1, -1, -1, 1, 1]
    
    numk, cx, cy = find_seeds(width, height, K, shape=shape)
    labels = -1 * np.ones((height, width), dtype=int)

    kl = np.zeros(numk)
    ka = np.zeros(numk)
    kb = np.zeros(numk)
    kx = np.zeros(numk)
    ky = np.zeros(numk)
    ksize = np.zeros(numk)

    # Priority queue
    heap = MinHeap()
    for k in range(numk):
        heap.insert((cx[k], cy[k], k), 0)

    CONNECTIVITY = 4
    #CONNECTIVITY = 8
    M = compactness
    invwt = (M * M * numk) / float(sz)

    while not heap.isEmpty():
        (x,y,k),_ = heap.pop()

        if labels[x,y] < 0:
            labels[x,y] = k

            kl[k] += data[x,y,0]
            ka[k] += data[x,y,1]
            kb[k] += data[x,y,2]

            kx[k] += x
            ky[k] += y
            ksize[k] += 1.0

            for p in range(CONNECTIVITY):
                xx = x + dx8[p]
                yy = y + dy8[p]
                if 0 <= xx < width and 0 <= yy < height:
                    if labels[xx,yy]<0:
                        lmean = kl[k] / ksize[k]
                        amean = ka[k] / ksize[k]
                        bmean = kb[k] / ksize[k]
                        xmean = kx[k] / ksize[k]
                        ymean = ky[k] / ksize[k]

                        ldiff = lmean - data[xx,yy,0]
                        adiff = amean - data[xx,yy,1]
                        bdiff = bmean - data[xx,yy,2]

                        xdiff = xmean - xx
                        ydiff = ymean - yy

                        colordist = ldiff**2 + adiff**2 + bdiff**2
                        spatialdist = xdiff**2 + ydiff**2
                        slicdist = colordist + invwt * spatialdist

                        heap.insert((xx,yy,k), slicdist)

    SPs = [[] for _ in range(numk)]
    for i in range(height):
        for j in range(width):
            SPs[labels[i,j]].append((i,j))
    return SPs


### proposed SNIC algorithm (made for gray-scale, RBG, and Hyperspectral images)
def mySNIC(data:np.ndarray, K:int,
           compactness:float=0,
           shape="square",
           simFun=norm2_similarity
           )->list[list[tuple[int,int]]]:
    """
    The proposed SNIC improvment algorithm for more images type, designed for hyperspectral images
    """
    height, width, B = data.shape
    sz = width * height
    dx8 = [-1, 0, 1, 0, -1, 1, 1, -1]
    dy8 = [0, -1, 0, 1, -1, -1, 1, 1]
    
    numk, cx, cy = find_seeds(width, height, K, shape=shape)
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
    M = compactness
    invwt = (M * M * numk) / float(sz)

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
                        colordist = simFun(means, data[xx,yy,:])

                        if compactness!=0:
                            xmean = kx[k] / ksize[k]
                            ymean = ky[k] / ksize[k]
                            xdiff = xmean - xx
                            ydiff = ymean - yy
                            spatialdist = xdiff**2 + ydiff**2
                            slicdist = colordist + invwt * spatialdist
                        else:
                            slicdist = colordist
                        heap.insert((xx,yy,k), slicdist)

    SPs = [[] for _ in range(numk)]
    for i in range(height):
        for j in range(width):
            SPs[labels[i,j]].append((i,j))
    return SPs
