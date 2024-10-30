import os
import random
import  matplotlib

import torch
import matplotlib.pyplot as plt
import pandas as pd
algos = ['cma','cmal','ig','adam','adagrad','adadelta']


def find_fL(problem,seed,algos):
    ds, net = problem
    bests = {}
    for algo in algos:
        file = 'history_'+algo+'_'+net+'_'+ds+'_seed_'+str(seed)+'.txt'
        stats = torch.load(file,map_location=torch.device('cpu'))
        bests[algo] = stats['train_loss'][-1]

    fL = min(bests.values())
    return fL, bests


def fw0(problem,seed):
    ds, net = problem
    algo = 'cma'
    file = 'history_' + algo + '_' + net + '_' + ds + '_seed_' + str(seed) + '.txt'
    stats = torch.load(file,map_location=torch.device('cpu'))
    fw0 = stats['train_loss'][0]
    return fw0


def c_sp(problem,tau,seed,algo,algos):
    ds, net = problem
    initial_loss = fw0(problem, seed)
    fL = find_fL(problem,seed,algos)[0]
    file = 'history_' + algo + '_' + net + '_' + ds + '_seed_' + str(seed) + '.txt'
    stats = torch.load(file,map_location=torch.device('cpu'))
    train_loss = stats['train_loss']
    elapsed_time = stats['time_4_epoch']
    csp = 1e8
    for idx,loss in enumerate(train_loss):
        if loss <= fL + tau*(initial_loss-fL):
            csp = elapsed_time[idx]
            break
    return csp

def r_sp(problem,tau,seed,solver):
    list_c = []
    c_ref = -1
    for algo in algos:
        csp = c_sp(algo,problem,tau,seed)
        if algo == solver:
            c_ref = csp
        list_c.append(csp)

    return c_ref/min(list_c)

def rho(solver,alpha,tau,seed,big_flag=False,small_flag=False):
    rh = 0
    assert (small_flag != big_flag or (small_flag == big_flag and big_flag == False))
    if big_flag == True:
        probs = [p for p in all_probs if p[1] in ['XXXL','4XL','XXL']]
    elif small_flag == True:
        probs = [p for p in all_probs if p[1] not in ['XXXL', '4XL', 'XXL']]
    else:
        probs = all_probs
    for problem in probs:
        rsp = r_sp(problem,tau,seed,solver)
        if rsp <= alpha:
            rh +=1
    return rh/len(probs)


def plot_PP(big_flag,small_flag,tau):
    plt.figure()
    plt.xscale('log')
    alphas = [1,2,4,8,16,32,64]
    seeds = [1,10,100,1000,10000]
    colors = ['red','blue','green','orange','black']
    mk = ['.','o','v']#,'*']
    idx = 0
    for solver in algos:
        pp = []
        for alpha in alphas:
            r = sum(rho(solver,alpha,tau,seed,big_flag,small_flag) for seed in seeds)/len(seeds)
            pp.append(r)
        plt.plot([_ for _ in alphas],pp,color=colors[idx],linestyle='solid',marker=mk[idx],markersize=7)
        idx +=1

    plt.xticks(ticks=alphas,labels=[str(_) for _ in alphas])
    plt.xlabel(chr(945))
    plt.ylabel(chr(961) + '(' + chr(945) + ')')
    plt.legend(algos)
    title = chr(964)+'='+str(tau)+' '
    if big_flag == True: title += '(Big networks)'
    if small_flag == True: title += '(Small networks)'
    plt.title(title)
    plt.savefig('PP_noLBFGS_'+title+'.pdf')
    print('DONE')
    #plt.show()

import numpy as np

def plotPP(seeds,tau,all_probs,big_flag,small_flag):
    list_R = []
    print(algos)
    if small_flag == True:
        all_probs = [p for p in all_probs if p[1] not in ['XXL','XXXL','4XL']]
    if big_flag == True:
        all_probs = [p for p in all_probs if p[1] in ['XXL','XXXL','4XL']]
    for seed in seeds:
        C = np.array([[c_sp(pr,tau,seed,algo,algos) for pr in all_probs] for algo in algos])
        c_star = np.min(C,0)
        R = C/c_star
        list_R.append(R)
    R = sum(list_R)/len(list_R)
    for i in range(len(algos)): R[i].sort()
    max_data = np.max(R[R<=100])
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i, j] > max_data:
                R[i, j] = 2 * max_data
    m = [pp/len(all_probs) for pp in range(1,len(all_probs)+1)]
    colors = ['b','r','g','orange','black','pink']
    plt.figure()
    plt.xlabel(chr(945))
    plt.ylabel(chr(961) + '(' + chr(945) + ')')
    plt.legend(algos)
    plt.xscale('log')
    plt.xlim(1,1.2*max_data)
    #plt.xticks((1,2,4,8))
    for i in range(len(algos)):
        plt.step(R[i],m,colors[i],linewidth=2.25)
    plt.legend(['CMA','CMA Light','IG','Adam','Adagrad','Adadelta'])
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.gca().ticklabel_format(style='plain')
    title = 'tau='+str(tau)+'_'
    if big_flag == True: title += '(BigNetworks)'
    if small_flag == True: title += '(SmallNetworks)'
    plt.title(title)
    #plt.show()
    plt.savefig('PP_'+title+'.pdf')
    print('DONE')
    return  C,R

if __name__ == '__main__':
    seeds = [1,10,100,1000,10000]
    taus = [1e-1,1e-2,1e-4]
    for tau in taus:
        dataset_list = [ds[:-4] for ds in os.listdir('dataset')]
        architectures = ['S', 'M', 'L', 'XL', 'XXL', 'XXXL', '4XL']
        all_probs = [(ds, net) for ds in dataset_list for net in architectures]
        big_datasets = ['BlogFeedback', 'Covtype', 'Protein', 'Skin NonSkin', 'YearPredictionMSD']
        os.chdir('prove_CMAL')
        c,r = plotPP(seeds,tau,all_probs,big_flag=False,small_flag=True)
        c, r = plotPP(seeds, tau, all_probs, big_flag=True, small_flag=False)