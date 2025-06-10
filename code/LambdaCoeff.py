import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import MyDataset as Data
from EntropyRateSuperpixel import find_superpixel

### Minimum Superpixel size
gamma = 0.15
def computePs(K, N, M):
    return int(N*M/K * 1/np.log(np.log(N*M/K)) * gamma)


### Find lambda coefficient methdos
def dichotomies_search(data, K, mini, maxi, similarity_function, Ps, print_info=False):
    def aux(i,j, SP):
        coeff = int((i+j)/2)
        if print_info:
            print("finding lambda coeff:", coeff)
        if i>=j:
            return SP, max(coeff,1)
        
        SP = find_superpixel(data, K, coeff, similarity_function)
        minSPsize = min([len(l) for l in SP])
        if minSPsize==Ps:
            return SP, coeff
        elif minSPsize<Ps:
            return aux(coeff+1, j, SP)
        else:
            return aux(i, coeff-1, SP)
        
    return aux(mini, maxi, None)


def findByCroping(data, simFun, nbCroping, n, m):
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
        plt.imshow(trainData[:,:,1])
        plt.show()
        Ps = computePs(K, N, M)
        _, coeff = dichotomies_search(trainData, K, a, b, simFun, Ps, True)
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

models = [model1, model2, model3, model4, model5]



def r2_score(z_true, z_pred):
        ss_res = np.sum((z_true - z_pred) ** 2)
        ss_tot = np.sum((z_true - np.mean(z_true)) ** 2)
        return 1 - ss_res / ss_tot



def find_best_models_parameters(usedSimFun, show_plot=False):
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

    


### Best models found
def getLambdaAverage(K,N,M):
    return 0.38 * gamma* K * np.log(N*M*K)**0.668

def getLambdaNorm2(K,N,M):
    return 1.214 * gamma* K * np.log(N*M*K)**0.444

def getLambdaNorm1(K,N,M):
    return 0.176* gamma* K * np.log(N*M)**1.147



def plot_lambda_models_vs_gt():
    LambdaCoeffs = Data.FoundLambdaCoeff["SimilarityFunction"]
    getLambdas = [getLambdaAverage, getLambdaNorm2, getLambdaNorm1]
    names = ["Basic", "(.)²", "|.|"]

    colors = ["orange", "mediumseagreen", "royalblue"]
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


#find_best_models_parameters(0, show_plot=True)
#plot_lambda_models_vs_gt()