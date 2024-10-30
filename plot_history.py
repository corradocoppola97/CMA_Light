import matplotlib.pyplot as plt
import os
import torch
#os.chdir('C:\\Users\\corra\\OneDrive\\Desktop\\prove_imclass')
algos = ['cma','cmal','adam','adagrad','adadelta']
networks = ['resnet18','resnet34','resnet50','resnet101','resnet152']#,'mobilenetv2']
seed = 1
dataset = 'cifar10'
colors = ['b','r','g','orange','black','pink']
labels = ['CMA','CMA Light','Adam','Adagrad','Adadelta']
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
    plt.xlim(0.95,2*max_data)
    #plt.xticks((1,2,4,8))
    m = [0.0] + m + [m[-1]]
    mk = ['*','.','o','v','s']
    for i in range(len(algos)):
        rr = list(R[i])
        #print(R[i])
        rr = [0.95] + rr + [5*max_data]
        plt.step(rr,m,colors[i],marker=mk[i],markersize=5)
    plt.legend(legend,loc='lower right')
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.gca().ticklabel_format(style='plain')
    print(R)
    print(m)
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
    n_ep = 51
    for j in range(n_ep):
        h1 = sum(histories[i]['train_loss'][j] for i in range(n))/n
        h2 = sum(histories[i]['val_accuracy'][j] for i in range(n)) / n
        h3 = sum(histories[i]['time_4_epoch'][j] for i in range(n)) / n
        train_loss.append(h1)
        val_accuracy.append(h2)
        eltime.append(h3)
    return train_loss, val_accuracy, eltime


"""seeds = [1,10,100,1000,10000]
all_probs = [('cifar10',n) for n in networks]
all_probs = all_probs + [('cifar100',n) for n in networks]

for tau in [1e-1,1e-2,1e-4]:
    plotPP_imclass(seeds,tau,all_probs,algos,legend=['CMA','CMA Light','Adam','Adagrad','Adadelta'])"""

# ds = 'cifar10'
# b = {}
# for algo in algos:
#     f = {}
#     for net in networks:
#         ba = get_accuracy(ds,algo,net,seeds)
#         f[net] = ba
#     b[algo] = f
#
# ds = 'cifar100'
# b2 = {}
# for algo in algos:
#     f = {}
#     for net in networks:
#         ba = get_accuracy(ds, algo, net, seeds)
#         f[net] = ba
#     b2[algo] = f


seeds = [1]
algos = ['cma','cmal','adam','adagrad']
for flag_epoch in [False]:
    for network in ['resnet18']:
        plt.figure()
        title = f'Training Loss - {network} on {dataset}'
        x_label = 'Epochs' if flag_epoch == True else 'Elapsed time'
        #if flag_epoch == True: plt.xlim(0,100)
        for i,algo in enumerate(algos):
            histories = []
            for seed in seeds:
                file = 'history_'+algo+'_'+network+'_'+dataset+'_seed_'+str(seed)+'.txt'
                history = torch.load(file)
                histories.append(history)
            train_loss, val_accuracy, eltime = get_avg_history(histories)
            if flag_epoch == True:
                y_axis = train_loss
                x_axis = [_ for _ in range(len(y_axis))]
            else:
                y_axis = train_loss
                x_axis = eltime
            plt.plot(x_axis,y_axis,color=colors[i],linewidth=1.35)
        plt.xlabel(x_label)
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend(labels)
        plt.savefig('Loss_'+dataset+'_'+network+'_'+x_label+'.pdf')
        plt.show()

        plt.figure()
        title = f'Test Accuracy - {network} on {dataset}'
        x_label = 'Epochs' if flag_epoch == True else 'Elapsed time'
        #if flag_epoch == True: plt.xlim(0, 100)
        for i, algo in enumerate(algos):
            histories = []
            for seed in seeds:
                file = 'history_' + algo + '_' + network + '_' + dataset + '_seed_' + str(seed) + '.txt'
                history = torch.load(file)
                histories.append(history)
            train_loss, val_accuracy, eltime = get_avg_history(histories)
            if flag_epoch == True:
                y_axis = val_accuracy
                x_axis = [_ for _ in range(len(y_axis))]
            else:
                y_axis = val_accuracy
                x_axis = eltime
            plt.plot(x_axis,y_axis,color=colors[i],linewidth=1.35)
        plt.xlabel(x_label)
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.legend(labels)
        plt.savefig('Accuracy_'+dataset+'_'+network+'_'+x_label+'.pdf')
        plt.show()
        print('Done')

