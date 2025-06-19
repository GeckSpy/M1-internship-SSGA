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
            100:[1, 17, 34],
            200:[1, 33, 72],
            300:[1, 52, 110],
            400:[1, 59, 150],
            500:[1, 78, 187],
            600:[1, 94, 230],
            700:[1, 133, 275]
        },
        PaviaUniversity["name"]:{
            100:[1, 17, 42],
            200:[1, 32, 84],
            300:[1, 47, 123],
            400:[1, 60, 166],
            500:[1, 81, 210],
            600:[1, 97, 251],
            700:[1, 111, 300]
        }
    },
    "SimilarityFunction":{
        # In order: average, norm2, norm1, perason
        IndianPines["name"]:{
            50: [17, 27, 15, 7],
            100:[34, 58, 31, 16],
            200:[72, 114, 70, 33],
            300:[110, 179, 102, 49],
            400:[150, 242, 141, 68],
            500:[187, 308, 180, 90],
            600:[230, 379, 217, 110]
        },
        PaviaUniversity["name"]:{
            100:[42, 59, 44, 16],
            200:[84, 121, 92, 34],
            300:[123, 185, 140, 51],
            400:[166, 255, 184, 68],
            500:[210, 331, 236, 85],
            600:[251, 404, 280, 105],
            700:[300, 481, 325, ]
        }
    }
}


PaviaCenterLambdaCoeff = {
    # In order: average, norm2, norm1, perason
    145:{
        100: [[50, 63, 51],
            [36, 60, 36],
            [40, 58, 38],
            [40, 62, 43],
            [36, 62, 40]],

        300: [[162, 238, 159],
            [121, 196, 118],
            [123, 200, 120],
            [125, 198, 139],
            [113, 219, 125]]
    },
    200:{
        100: [[38, 60, 40],
            [40, 60, 40],
            [38, 60, 43],
            [38, 62, 46]],

        300: [[123, 200, 120],
            [125, 186, 121],
            [123, 195, 133],
            [118, 220, 152]]
    },
    400:{
        100: [[44, 66, 47],
            [46, 58, 46],
            [46, 64, 46]],
        300: [[130, 196, 146],
            [144, 180, 142],
            [144, 219, 149]]
    },
    600:{
        100: [[49, 62, 46]],
        300: [[144, 181, 144]]
    }
}

for size, dic in PaviaCenterLambdaCoeff.items():
    for K in dic.keys():
        dic[K] = np.array(dic[K])
