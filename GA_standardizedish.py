import numpy as np
import data_parser
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.kernel_ridge import KernelRidge
import random


X=["descriptor1", "descriptor2"' ...']
Y1="response variable"

datapath="training data path"
testdatapath='testing data path
savepath='{}.png'

data = data_parser.parse(datapath)
data.set_x_features(X)
data.set_y_feature(Y1)
data.remove_all_filters()
# add filters for training data

lwrdata = data_parser.parse(lwrdatapath)
lwrdata.set_x_features(X)
lwrdata.set_y_feature(Y2)
lwrdata.remove_all_filters()
# add filters for testing data

Ydata = data.get_y_data()
Xdata = data.get_x_data()
Ydata_test = testdata.get_y_data()
Xdata_test = testdata.get_x_data()


# define k-fold cv test
def kfold_cv(model, X, Y, num_folds=5, num_runs=10):
    Xdata = np.asarray(X)
    Ydata = np.asarray(Y)
    #print(np.shape(Xdata), np.shape(Ydata))
    
    Krms_list = []
    for n in range(num_runs):
        kf = cross_validation.KFold(len(Xdata), n_folds=num_folds, shuffle=True)
        K_fold_rms_list = []
        # split into testing and training sets
        for train_index, test_index in kf:
            #print(np.shape(train_index), np.shape(test_index))
            X_train, X_test = Xdata[train_index], Xdata[test_index]
            Y_train, Y_test = Ydata[train_index], Ydata[test_index]
            # train on training sets
            model.fit(X_train, Y_train)
            Y_test_Pred = model.predict(X_test)
            Krms = np.sqrt(mean_squared_error(Y_test, Y_test_Pred))
            K_fold_rms_list.append(Krms)

        Krms_list.append(np.mean(K_fold_rms_list))

    return (np.mean(Krms_list))

# define extrapolation test
def TestData_extrapolation (model, Xtrain, Ytrain, Xtest, Ytest ):
    Xtrain = np.asarray(Xtrain)
    Ytrain = np.asarray(Ytrain)
    Xtest = np.asarray(Xtest)
    Ytest = np.asarray(Ytest)
    model.fit(Xtrain, Ytrain)
    Ypredict = model.predict(Xtest)
    rms = np.sqrt(mean_squared_error(Ypredict, Ytest))
    return (rms)

def generation(pop, num_parameters, num_parents, crossover_prob, mutation_prob, shift_prob):

    rmsList = []
    params = []
    for ind in range(len(pop)):
        
        newX_Train = np.copy(Xdata)
        newX_Test = np.copy(Xdata_test)
        params = pop[ind]
        # uses the numbers (except the last 2) in params to scale each descriptor (this is if you are using scaling parameters)
        for gene in range(len(params)-2):
            newX_Train[:, gene] = newX_Train[:, gene] * params[gene]
            newX_Test[:, gene] = newX_Test[:, gene] * params[gene]

        # puts the last two numbers in params into an equation to convert their ranges so they can be used for alpha and gamma
        model = KernelRidge(alpha = 10**(float(params[-2])*(-6/10)), gamma = 10**((float(params[-1])*(3/10))-1.5), kernel = 'rbf')

        # un-comment one of these lines to select which test to use
        #rms = kfold_cv(model, newX_Train, Ydata, num_folds = 5, num_runs = 10)
        #rms = TestData_extrapolation(model, newX_Train, Ydata, newX_Test, Ydata_LWR)
        rmsList.append(rms)

    #select parents
    parents = np.empty([num_parents, num_parameters])
    parentRMS = []
    for newP in range(num_parents):       
        parents[newP] = pop[np.argmin(rmsList)]
        parentRMS.append(np.min(rmsList))
        rmsList[np.argmin(rmsList)] = np.max(rmsList)

    #progenate new population 
    for ind in range(len(pop)):
        p1 = parents[random.randrange(0, num_parents)]
        p2 = parents[random.randrange(0, num_parents)]
        #print(p1, p2)
        
        for par in range(num_parameters):
            p = random.random()
            m = random.random()
            s = random.random()
            #print(p, m, s)
            if p <= crossover_prob:
                pop[ind][par] = p1[par]
            else:
                pop[ind][par] = p2[par]
            if (s <= shift_prob):
                #pop[ind][par] = np.abs( pop[ind][par] + (random.random() - .5)*1 )      # use this line for random shift when using no increments
                pop[ind][par] = np.abs( pop[ind][par] + random.randrange(-4,4)/10 )     # use this line for random shift when using incerenemts of .1
            if m <= mutation_prob:
                #pop[ind][par] = random.random()*10          # use this line for mutation when using no incerenemts
                pop[ind][par] = random.randrange(0,100)/10   # use this line for mutation when using increments of .1 

    return { 'new_population':pop, 'best_parameters':parents, 'best_rms':parentRMS }

numParams = 11 # number of parameters to optimize
#initialize empty population with 50 individuals
population = np.empty([50,numParams])

bestRMSs = []
bestGens = []
allBestParams = []
runs=0
runsSinceBest = 0
numRuns = 5   
print('running')
while runs < numRuns: # do 5 runs 
    #print(runs)
    runsSinceBest = 0
    runs = runs+1
    gens = 1
    bestRMS = 200
    bestParams = []
    
    #randomize population
    for individual in range(len(population)):
        for gene in range(len(population[individual])):
            #population[individual][gene]  = random.random()*10              # use this line when using no increments
            population[individual][gene]  = random.randrange(0, 100)/10      # use this line when using increments of .1 

    
    while runsSinceBest < 30 and gens < 200: # runs until there hasn't been an improvement in 30 generations, or has been running for 200 total generations
        
        # run a generation imputting population, number of parameters, number of parents, crossover probablility, mutation probability, random shift probability
        Gen = generation(population, numParams, 10, .5, .1, .5 )
        print(Gen['best_rms'])
        print(Gen['best_parameters'])
        population = Gen['new_population']
        
        if bestRMS > np.min(Gen['best_rms']) :
            bestRMS = np.min(Gen['best_rms'])
            bestParams = Gen['best_parameters'][np.argmin(Gen['best_rms'])]            
            bestRMSs.append(bestRMS)
            bestGens.append(gens)
            runsSinceBest = 0

        bestParamsSTR = ' '
        for i in range(len(bestParams)):
            bestParamsSTR = bestParamsSTR+"{0:.2f}".format(bestParams[i])+" , "     # rounds each parameter in list to the nearest hundreth for convenience
        print(bestParamsSTR)
        print(bestRMS)   
        print(runs, gens, runsSinceBest)
        gens = gens+1
        runsSinceBest = runsSinceBest +1
    allBestParams.append(bestParams)
    allBestRMS.append(bestRMS)

avgParams = zeros(numParams)
for i in range(len(allBestParams)):
    bestParamsSTR = ' '
    for n in range(numParams):
        bestParamsSTR = bestParamsSTR+"{0:.2f}".format(allBestParams[i][n])+" , "   # rounds each parameter in list to the nearest hundreth for convenience
        avgParams[n] = avgParams[n] + allBestParams[i][n]                           # sums each param over 5 runs to calculate average for each
    print(bestParamsSTR+str(allBestRMS[i]))

print(np.mean(allBestRMS), np.std(allBestRMS))

avgParamsSTR = ' '
for n in range(numParams):
    avgParams[n] = avgParams[n]/numRuns
    avgParamsSTR = avgParamsSTR + "{0:.2f}".format(avgParams[n])+" , "      # rounds each parameter in list to the nearest hundreth for convenience
print(avgParamsSTR)



