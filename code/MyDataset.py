import numpy as np
import scipy.io
import json
from sklearn.preprocessing import StandardScaler

datasets_folder = "datasets/"

def classes(gt:np.ndarray,
            dic:dict[int, str]
            ) -> dict[int, tuple[str, list[tuple[int,int]]]]:
    """
    Return the list of pixels indices corresponding to each classes
    """
    res = {}
    for key,value in dic.items():
        res[key] = [value, []]

    n,m = gt.shape
    for i in range(n):
        for j in range(m):
            res[gt[i,j]][1].append((i,j))
    return res



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


def standardize_data(data:np.ndarray):
    """
    Standardize the data
    """
    N,M,B = data.shape
    new_data = data.reshape(N*M, B)
    return StandardScaler().fit(new_data).transform(new_data).reshape(N,M,B)


def load_dataset(path :str,
                 data_class: dict[int, str],
                 name :str,
                 data_key :str=None,
                 gt_key :str=None,
                 gt_path :str=None):
    """
    Return the dataset in my format
    """
    if gt_path==None:
        gt_path = path + "_gt"
    if data_key==None:
        data_key = path[0].lower() + path[1:]
    if gt_key==None:
        gt_key = gt_path[0].lower() + gt_path[1:]

    data = scipy.io.loadmat(datasets_folder + path +".mat")[data_key]
    gt = scipy.io.loadmat(datasets_folder + gt_path + ".mat")[gt_key]

    data_classes = classes(gt, data_class)
    dataset = {
        "name": name,
        "shape": data.shape,
        "gt": gt,
        "data": data,#normalized_data(data),
        "class": data_classes,
        "labels": [i for i in data_classes.keys() if i!=0]
    }
    return dataset




def save_dataset(name:str, dic:dict):
    dic_to_save = dic.copy()
    dic_to_save["gt"] = dic["gt"].tolist()
    dic_to_save["data"] = dic["data"].tolist()
    with open(datasets_folder + name +".json", "w") as f:
         f.write(json.dumps(dic_to_save))



### Indian Pines
IP_class = {0:"Vegetation",
            1:"Alfalfa",
            2:"Corn-notill",
            3:"Corn-mintill",
            4:"Corn",
            5:"Grass-pasture",
            6:"Grass-trees",
            7:"Grass-pasture-mowed",
            8:"Hay-windrowed",
            9:"Oats",
            10:"Soybean-notill",
            11:"Soybean-mintill",
            12:"Soybean-clean",
            13:"Wheat",
            14:"Woods",
            15:"Buildings-Grass-Trees-Drives",
            16:"Stone-Steel-Towers"
}
IndianPines = load_dataset("Indian_pines_corrected", IP_class, "Indian Pines", gt_path="Indian_pines_gt")
#IndianPines["data"] = IndianPines["data"][:,:,1:]
#save_dataset("indian_pines", IndianPines)


### Pavia University
PU_class = {0:"No information",
            1:"Asphalt", 
            2:"Meadows",
            3:"Gravel",
            4:"Trees",
            5:"Painted metal sheets",
            6:"Bare Soil",
            7:"Bitumen",
            8:"Self-Blocking Bricks",
            9:"Shadows"
}
PaviaUniversity = load_dataset("PaviaU", PU_class, "Pavia University")


### Pavia Center
PaviaCenter = load_dataset("Pavia", PU_class, "Pavia Center")
