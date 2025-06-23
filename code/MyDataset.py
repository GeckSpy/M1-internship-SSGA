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
        "data": data,
        "class": data_classes,
        "labels": [i for i in data_classes.keys()]
    }
    return dataset




def save_dataset(name:str, dic:dict):
    dic_to_save = dic.copy()
    dic_to_save["gt"] = dic["gt"].tolist()
    dic_to_save["data"] = dic["data"].tolist()
    with open(datasets_folder + name +".json", "w") as f:
         f.write(json.dumps(dic_to_save))



### Indian Pines
IP_class = {0:"NoInfo",
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
PU_class = {0:"NoInfo",
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








#### Lambda coefficient (found thanks to LambdaCoeff.dichotomies_search)
FoundLambdaCoeff = {
    "DataFormat":{
        # In order: standard, normalized, standardized
        IndianPines["name"]:{
            100:[1, 17, 38],
            200:[1, 33, 76],
            300:[1, 46, 114],
            400:[1, 61, 161],
            500:[1, 79, 202],
            600:[1, 97, 254],
            700:[1, 119, 296]
        },
        PaviaUniversity["name"]:{
            100:[1, 16, 49],
            200:[1, 33, 99],
            300:[1, 47, 149],
            400:[1, 63, 201],
            500:[1, 81, 252],
            600:[1, 86, 299],
            700:[1, 111, 359]
        }
    },
    "SimilarityFunction":{
        # In order: average, norm2, norm1, perason
        IndianPines["name"]:{
            50: [17, 26, 16, 9],
            100:[38, 53, 30, 17],
            200:[76, 114, 73, 40],
            300:[114, 170, 108, 55],
            400:[161, 232, 152, 75],
            500:[202, 305, 191, 103],
            600:[254, 372, 232, 129]
        },
        PaviaUniversity["name"]:{
            100:[49, 59, 50, 20],
            200:[99, 121, 101, 40],
            300:[149, 188, 153, 62],
            400:[201, 257, 201, 83],
            500:[252, 331, 253, 103],
            600:[299, 410, 304, 126],
            700:[359, 493, 356, 148]
        }
    }
}


PaviaCenterLambdaCoeff = {
    # In order: average, norm2, norm1, perason
    145:{
        100: [[50, 65, 52, 16],
            [48, 62, 48, 17],
            [41, 62, 46, 21],
            [48, 75, 52, 19]],

        300: [[168, 237, 179, 55],
            [171, 234, 169, 44],
            [139, 210, 144, 69],
            [130, 208, 181, 59]]
    },
    200:{
        100: [[46, 61, 48, 19],
            [46, 56, 46, 16],
            [44, 59, 46, 22]],

        300: [[148, 197, 160, 66],
            [148, 199, 150, 53],
            [146, 191, 150, 69]]
    },
    400:{
        100: [[46, 66, 50, 21],
            [44, 58, 48, 21]],
        300: [[150, 195, 148, 66],
            [139, 194, 155, 64]]
    },
    600:{
        100: [[48, 61, 48, 19]],
        300: [[148, 204, 148, 59]]
    }
}

for size, dic in PaviaCenterLambdaCoeff.items():
    for K in dic.keys():
        dic[K] = np.array(dic[K])
