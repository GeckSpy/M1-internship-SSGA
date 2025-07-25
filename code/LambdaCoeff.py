import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import MyDataset as Data
from EntropyRateSuperpixel import find_superpixel, getLambdaAverage, getLambdaNorm1, getLambdaNorm2, getLambdaPerason

### Minimum Superpixel size
gamma = 0.15
def computePs(K:int, N:int, M:int):
    """
    Compute the minimal allowed size for SPS
    - K: number of wanted superpixels
    - N, M: dimension of the image
    """
    return int(N*M/K * 1/np.log(np.log(N*M/K)) * gamma)



### Find lambda coefficient methdos
def dichotomies_search(data:np.ndarray,
                       K:int,
                       mini:int, maxi:int,
                       similarity_function,
                       Ps:int,
                       print_info:bool=False):
    """
    Find the best lambda coefficient based on the minimal SP size criteria with a dichotomies search
    """
    def aux(i,j, SP):
        coeff = int((i+j)/2)
        if print_info:
            print("finding lambda coeff:", coeff)
        if i>=j:
            return SP, max(coeff,1)
        
        SP = find_superpixel(data, K, 
                             lambda_coef=coeff,
                             simFun=similarity_function)
        minSPsize = min([len(l) for l in SP])
        if minSPsize==Ps:
            return SP, coeff
        elif minSPsize<Ps:
            return aux(coeff+1, j, SP)
        else:
            return aux(i, coeff-1, SP)
        
    return aux(mini, maxi, None)



def findByCroping(data:np.ndarray,
                  similarity_function,
                  nbCroping:int,
                  n:int, m:int,
                  do_plot:bool=False):
    """
    Try to find a good lambda coefficient by taking average of lambda coefficient of cropped part of the image
    - n,m: dimension of the cropped imgs
    """
    # Didn't performmed well
    N,M,B = data.shape
    sum_lambda = 0

    K = 100
    a = int(20*K/100)
    b = int(60*K/100)
    for _ in range(nbCroping):
        xStart = np.random.randint(0, N-1-n)
        yStart = np.random.randint(0, M-1-m)
        trainData = Data.standardize_data(data)[xStart:xStart+n, yStart:yStart+m, :]
        if do_plot:
            plt.imshow(trainData[:,:,1])
            plt.show()
        Ps = computePs(K, N, M)
        _, coeff = dichotomies_search(trainData, K, a, b, similarity_function, Ps, True)
        print(coeff)
        sum_lambda += coeff
    return sum_lambda/nbCroping



### Find best model
def model1(xy, a, b):
    x,y = xy
    return a*y* gamma* np.log(x)**b

def model2(xy, a, b):
    x,y= xy
    return a * gamma* y *np.log(x/y)**b

def model3(xy, a, b):
    x,y= xy
    return a * gamma * y *np.log(np.log(x/y))**b

def model4(xy, a, b):
    x,y = xy
    return a * gamma* y * np.log(x*y)**b

def model5(xy, a, b):
    x,y = xy
    return a * gamma *y * np.log(np.log(x*y))**b

def model6(xy, a, b, c):
    x,y = xy
    return a * gamma* y**b * np.log(x*y)**c


models = [model1, model2, model3, model4, model5, model6]


def r2_score(z_true, z_pred):
    """
    Compute the R² score
    """
    ss_res = np.sum((z_true - z_pred) ** 2)
    ss_tot = np.sum((z_true - np.mean(z_true)) ** 2)
    return 1 - ss_res / ss_tot



def find_best_models_parameters(simFunName, show_plot:bool=False):
    usedSimFun = {"average":0, "norm2":1, "norm1":2, "perason":3}[simFunName]
    LambdaCoeffs = Data.FoundLambdaCoeff["SimilarityFunction"]
    points = []
    for dataset in [Data.IndianPines, Data.PaviaUniversity]:
        n, m, _ = dataset["data"].shape
        for K in LambdaCoeffs[dataset["name"]].keys():
            points.append((n*m, K, LambdaCoeffs[dataset["name"]][K][usedSimFun]))

    for size, dic in Data.PaviaCenterLambdaCoeff.items():
        for K in dic.keys():
            point = (size**2, K, np.average(dic[K][:,usedSimFun]))
            points.append(point)

    points = np.array(points)
    xy = (points[:,0], points[:,1])
    z = points[:,2]


    popts = []
    for i,fun in enumerate(models):
        popt, _ = curve_fit(fun, xy, z)
        popts.append(popt)

        z_pred = models[i](xy, *popts[i])
        print("model n°" + str(i+1)+":")
        print("   ", "R² score:", r2_score(z, z_pred))
        print("   ", "params:", *popts[i])


    if show_plot:
        fig, axs = plt.subplots(1, 2)
        for id,dataset in enumerate([Data.IndianPines, Data.PaviaUniversity]):
            n, m, _ = dataset["data"].shape
            new_Ks = LambdaCoeffs[dataset["name"]].keys()
            axs[id].plot(new_Ks, [LambdaCoeffs[dataset["name"]][K][usedSimFun] for K in new_Ks], "-o", label="gt")
            
            for i, fun in enumerate(models):
                x_temp = np.array([n*m for K in new_Ks])
                y_temp = np.array([K for K in new_Ks])
                xy_temp = (x_temp, y_temp)
                axs[id].plot(new_Ks, models[i](xy_temp, *popts[i]), ":x", label="model "+str(i+1))

            axs[id].title.set_text(dataset["name"])
            axs[id].set_xlabel("K")
            axs[id].set_ylabel("Lambda")

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(models)+1, bbox_to_anchor=(0.5, -0.0))
        fig.suptitle("Lambda coefficients models", fontsize=14)
        plt.tight_layout() 
        plt.subplots_adjust(bottom=0.20)
        plt.show()

    


def plot_lambda_models_vs_gt():
    """
    Plot the lambdas models compared to the found values of lambda
    """
    LambdaCoeffs = Data.FoundLambdaCoeff["SimilarityFunction"]
    getLambdas = [getLambdaAverage, getLambdaNorm2, getLambdaNorm1, getLambdaPerason]
    names = ["Basic", "norm2", "norm1", "perason"]

    colors = ["orange", "mediumseagreen", "royalblue", "violet"]
    fig, axs = plt.subplots(1, 2)
    for id,dataset in enumerate([Data.IndianPines, Data.PaviaUniversity]):
        n, m, _ = dataset["data"].shape
        new_Ks = LambdaCoeffs[dataset["name"]].keys()
        for i in range(len(getLambdas)):
            axs[id].plot(new_Ks, [LambdaCoeffs[dataset["name"]][K][i] for K in new_Ks], "-o", label=names[i], color=colors[i])
            axs[id].plot(new_Ks, [getLambdas[i](K, n, m) for K in new_Ks], "--x", label=names[i]+"'s model", color=colors[i])
        axs[id].title.set_text(dataset["name"])
        axs[id].set_xlabel("K")
        axs[id].set_ylabel("Lambda")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(models)+1, bbox_to_anchor=(0.5, -0.0))
    plt.tight_layout() 
    plt.subplots_adjust(bottom=0.20)
    plt.show()


#find_best_models_parameters("norm2", show_plot=True)
#plot_lambda_models_vs_gt()