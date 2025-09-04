import matplotlib.pyplot as plt
import os
import torch
#os.chdir('C:\\Users\\corra\\OneDrive\\Desktop\\prove_imclass')
algos = ['sgd','cmal','adam','adagrad','adadelta']
networks = ['resnet18','resnet34','resnet50','resnet101','resnet152']#,'mobilenetv2']
seed = 1
dataset = 'cifar10'
colors = ['b','r','g','orange','black','pink']
labels = ['SGD','Adam','CMA Light','Adagrad','Adadelta']
from PerformanceProfiles import c_sp
import numpy as np
import matplotlib



def plotPP_imclass(seeds,tau,all_probs,algos,legend):
    list_R = []
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
                R[i, j] = 1.2 * max_data
    m = [pp/len(all_probs) for pp in range(1,len(all_probs)+1)]
    colors = ['b','r','g','orange','black','pink']
    plt.figure()
    plt.xlabel(chr(945))
    plt.ylabel(chr(961) + '(' + chr(945) + ')')
    plt.legend(algos)
    plt.xscale('log')
    #plt.xlim(0.95,1.75)
    #plt.xticks((1,2,4,8))
    m = [0.0] + m + [m[-1]]
    mk = ['*','.','o','v','s']
    for i in range(len(algos)):
        rr = list(R[i])
        #print(R[i])
        rr = [0.95] + rr + [2*max_data]
        plt.step(rr,m,colors[i],marker=mk[i],markersize=5)
    plt.legend(legend,loc='lower right')
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.gca().ticklabel_format(style='plain')
    title = 'tau='+str(tau)+''
    plt.title(title)
    #plt.show()
    plt.savefig('PP_'+title+'.pdf')
    plt.show()
    print('DONE')
    return  C,R


def get_accuracy(ds,algo,net,seeds):
    best_accs = []
    for i in range(len(seeds)):
        file = 'history_'+algo+'_'+net+'_'+ds+'_seed_'+str(seeds[i])+'.txt'
        stats = torch.load(file)
        best_acc = max(stats['val_accuracy'])
        best_accs.append(best_acc)
    return best_accs


def get_avg_history(histories):
    n = len(histories)
    train_loss, val_accuracy, eltime = [], [], []
    n_ep = 40
    for j in range(n_ep):
        h1 = sum(histories[i]['train_loss'][j] for i in range(n))/n
        h2 = sum(histories[i]['val_accuracy'][j] for i in range(n)) / n
        h3 = sum(histories[i]['time_4_epoch'][j] for i in range(n)) / n
        train_loss.append(h1)
        val_accuracy.append(h2)
        eltime.append(h3)
    return train_loss, val_accuracy, eltime


seeds = [1,10,100,1000,10000]
all_probs = [('cifar10',n) for n in networks]
all_probs = all_probs + [('cifar100',n) for n in networks]

for tau in [0.5,0.25,0.1,0.01]:
    plotPP_imclass(seeds,tau,all_probs,algos,legend=['SGD','COFF_IG','Adam','Adagrad','Adadelta'])
