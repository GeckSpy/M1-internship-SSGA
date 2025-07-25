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

    key_to_del = []
    for key,value in res.items():
        if len(value[1])==0:
            key_to_del.append(key)
    for key in key_to_del:
        del res[key]
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
                 dataset_name :str,
                 data_key :str=None,
                 gt_key :str=None,
                 gt_path :str=None,
                 merges = None):
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

    if merges!=None:
        new_labels = {l:[l,v] for l,v in data_class.items()}
        for liste, name in merges:
            repr = min(liste)
            for id in liste:
                new_labels[id][0]=repr
                new_labels[id][1]=name
        data_class = {val[0]:val[1] for val in new_labels.values()}

        N,M = gt.shape
        new_gt = np.zeros(gt.shape, dtype=int)
        for i in range(N):
            for j in range(M):
                new_gt[i,j] = new_labels[gt[i,j]][0]
        gt = new_gt
    

    data_classes = classes(gt, data_class)


    dataset = {
        "name": dataset_name,
        "shape": data.shape,
        "gt": gt,
        "data": data,
        "class": data_classes,
        "labels": [i for i in data_classes.keys()],
        "data_class": data_class
    }
    return dataset


def copyDataset(dataset):
    return {
        "name": dataset["name"],
        "shape": dataset["data"].shape,
        "gt": dataset["gt"].copy(),
        "data": dataset["data"].copy(),
        "class": classes(dataset["gt"], dataset["data_class"]),
        "labels": [l for l in dataset["labels"]],
        "data_class": dataset["data_class"]
    }


def cropDataset(dataset, sizeX, sizeY):
    dataset = copyDataset(dataset)
    N,M,_ = dataset["shape"]
    sizeX, sizeY = min(sizeX, N), min(sizeY, M)
    xStart = np.random.randint(0, N-1-sizeX)
    yStart = np.random.randint(0, M-1-sizeY)

    dataset["data"] = dataset["data"][xStart:xStart+sizeX, yStart:yStart+sizeY, :]
    dataset["gt"] = dataset["gt"][xStart:xStart+sizeX, yStart:yStart+sizeY]
    dataset["shape"] = dataset["data"].shape
    dataset["class"] = classes(dataset["gt"], dataset["data_class"])
    dataset["labels"] = [i for i in dataset["class"].keys()]

    return dataset, xStart, yStart



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
#save_dataset("indian_pines", IndianPines)

IPmerges = [[[2,3,4],"Corn"], [[5,7],"Grass-Pasture"], [[10,11,12],"Soybean"]]
IndianPinesMerged = load_dataset("Indian_pines_corrected", IP_class, "Indian Pines Merged", gt_path="Indian_pines_gt",
                                 merges=IPmerges)






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


### Salinas Scene
SS_class = {0:"NoInfo",
            1:"Brocoli green weeds 1",
            2:"Brocoli green weeds 2",
            3:"Fallow",
            4:"Fallow rough plow",
            5:"Fallow smooth",
            6:"Stubble",
            7:"Celery",
            8:"Grapes_untrained",
            9:"Soil_vinyard_develop",
            10:"Corn senesced green weeds",
            11:"Lettuce romaine-4wk",
            12:"Lettuce romaine-5wk",
            13:"Lettuce romaine-6wk",
            14:"Lettuce romaine-7wk",
            15:"Vinyard untrained",
            16:"Vinyard vertical trellis"
}
SalinasScene = load_dataset("Salinas_corrected", SS_class, "Salinas Scene", gt_path="Salinas_gt")

SSmerges = [[[1,2,3],"Brocoli"], [[3,4,5],"Fallow"], [[11,12,13,14],"Lettuce romaine"], [[15,16], "Vinyard"]]
SalinasSceneMerged = load_dataset("Salinas_corrected", SS_class, "Salinas Scene Merged", gt_path="Salinas_gt",
                                 merges=SSmerges)





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



### SLIC, SNIC, ERS hyperparameter value
hyperparameterValue = {
    IndianPines["name"]:{
        50: [30, 0.005, 8],
        100: [30, 0.005, 17],
        200: [20, 0.003, 33],
        300: [20, 0.003, 52],
        400: [15, 0.002, 70],
        500: [10, 0.001, 88],
        600: [10, 0.001, 106]
    },
    SalinasScene["name"]:{
        20:[10, 0.001, 4],
        50:[8, 0.0005, 8],
        100:[8, 0.0005, 16],
        200:[3, 0.0001, 31],
        300:[1, 0.0001, 47],
        500:[1, 0.0001, 67],
        700:[1, 0.0001, 104],
        900:[1, 0.0001, 122],
    },
    PaviaUniversity["name"]:{
        100:[0,0, 15],
        300:[],
        500:[],
        700:[],
        900:[],
        1100:[],
        1300:[],
        1500:[]
    }
}