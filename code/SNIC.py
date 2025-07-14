import numpy as np
import heapq
from skimage import color
from classes import MinHeap


def rgb_to_lab_image(img):
    """
    Converts an RGB image to LAB using skimage.
    Expects image as numpy array with shape (H, W, 3), dtype uint8.
    """
    img_float = img.astype(np.float32) / 255.0
    lab = color.rgb2lab(img_float)
    return lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]  # L, A, B


def find_seeds(width, height, numk):
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



def run_snic(lv, av, bv, width, height, innumk, compactness):
    sz = width * height
    dx8 = [-1, 0, 1, 0, -1, 1, 1, -1]
    dy8 = [0, -1, 0, 1, -1, -1, 1, 1]
    dn8 = [-1, -width, 1, width, -1 - width, 1 - width, 1 + width, -1 + width]
    
    numk, cx, cy = find_seeds(width, height, innumk)
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
                        ldiff = kl[k] - lv[ii] * ksize[k]
                        adiff = ka[k] - av[ii] * ksize[k]
                        bdiff = kb[k] - bv[ii] * ksize[k]
                        xdiff = kx[k] - xx * ksize[k]
                        ydiff = ky[k] - yy * ksize[k]

                        colordist = ldiff**2 + adiff**2 + bdiff**2
                        xydist = xdiff**2 + ydiff**2
                        slicdist = (colordist + xydist * invwt) / (ksize[k]**2)

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




def snic_segmentation(image, num_superpixels=100, compactness=10.0):
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
    
    labels, numk = run_snic(lv, av, bv, width, height, num_superpixels, compactness)
    return labels, numk

