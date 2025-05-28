import numpy as np
import scipy.io
import json

datasets_folder = "datasets/"

def classes(gt, dic:dict):
    res = {}
    for key,value in dic.items():
        res[key] = [value, []]

    n,m = gt.shape
    for i in range(n):
        for j in range(m):
            res[gt[i,j]][1].append((i,j))
    return res



def normalized_data(data:np.ndarray):
    N,M,B = data.shape
    res = np.zeros(data.shape)
    for b in range(B):
        img = data[:,:,b]
        res[:,:,b] = (img - img.min())/(img.max()-img.min())
    return res

def load_dataset(path:str, data_class, name, data_key=None, gt_key=None, gt_path=None):
    if gt_path==None:
        gt_path = path + "_gt"
    if data_key==None:
        data_key = path[0].lower() + path[1:]
    if gt_key==None:
        gt_key = gt_path[0].lower() + gt_path[1:]



    data = scipy.io.loadmat(datasets_folder + path +".mat")[data_key]
    gt = scipy.io.loadmat(datasets_folder + gt_path + ".mat")[gt_key]

    dataset = {
        "name": name,
        "shape": data.shape,
        "gt": gt,
        "data": normalized_data(data),
        "class": classes(gt, data_class)
    }

    return dataset




def save_dataset(name, dic:dict):
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

### Paavia University
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


PaviaCenter = load_dataset("Pavia", PU_class, "Pavia Center")