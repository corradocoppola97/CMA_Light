import os
import torch
import matplotlib.pyplot as plt
from torchsummaryX import summary
import csv
import ast
import torchvision
from typing import Union
from cmalight import CMA_L
from cma import CMA
import torch.nn as nn



#Used to compute the loss function over the entire data set
def closure(data_loader,
            model,
            criterion,
            device):
    loss = 0
    with torch.no_grad():
        K = 1024
        for x,y in data_loader:
            x,y = x.to(device), y.to(device)
            y_pred = model(x)
            batch_loss = criterion(y_pred, y)
            loss += batch_loss.item()*(len(x)/K)
    return loss

#Used to compute the accuracy over the entire data set
def accuracy(data_loader: torch.utils.data.DataLoader,
            model: torchvision.models,
            device: Union[torch.device,str]):
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions / total_samples
    return accuracy

def get_w(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_w(model, w):
    index = 0
    for param in model.parameters():
        param_size = torch.numel(param)
        param.data = w[index:index+param_size].view(param.size()).to(param.device)
        index += param_size


def set_optimizer(opt:str, model: torchvision.models, *args, **kwargs):
    if opt == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(),*args,**kwargs)
    elif opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),*args,**kwargs)
    elif opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),*args,**kwargs)
    elif opt == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(),*args,**kwargs)
    elif opt == 'adamax':
        optimizer = torch.optim.Adamax(model.parameters(),*args,**kwargs)
    elif opt == 'asgd':
        optimizer = torch.optim.ASGD(model.parameters(),*args,**kwargs)
    elif opt == 'nadam':
        optimizer = torch.optim.NAdam(model.parameters(),*args,**kwargs)
    elif opt == 'radam':
        optimizer = torch.optim.RAdam(model.parameters(),*args,**kwargs)
    elif opt == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(),*args,**kwargs)
    elif opt == 'rprop':
        optimizer = torch.optim.Rprop(model.parameters(),*args,**kwargs)
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),*args,**kwargs)
    elif opt == 'cmal':
        optimizer = CMA_L(model.parameters(),verbose=True,verbose_EDFL=True)
    elif opt == 'cma':
        optimizer = CMA(model.parameters(),verbose=True,verbose_EDFL=True)
    else:
        raise SystemError('RICORDATI DI SCEGLIERE L OTTIMIZZATORE!  :)')
    return optimizer

def hardware_check():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Actual device: ", device)
    if 'cuda' in device:
        print("Device info: {}".format(str(torch.cuda.get_device_properties(device)).split("(")[1])[:-1])

    return device

def print_model(model, device, save_model_root, input_shape):
    info = summary(model, torch.ones((1, 3, 32, 32)).to(device))
    info.to_csv(save_model_root + 'model_summary.csv')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




def plot_graph(data, label, title, path):
    epochs = range(0, len(data))
    plt.plot(epochs, data, 'orange', label=label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid('on', color='#cfcfcf')
    plt.tight_layout()
    plt.savefig(path + title + '.pdf')
    plt.close()


def plot_history(history, path):
    plot_graph(history['train_loss'], 'Train Loss', 'Train_loss', path)
    plot_graph(history['val_acc'], 'Val Acc.', 'Val_acc', path)


def extract_history(history_file):
    with open(history_file) as file_obj:
        reader_obj = csv.reader(file_obj)
        for row in reader_obj:
            l = row[0]
            break
        l = l[:-9]
        train_loss = ast.literal_eval(l)
        val_accuracy = row[498:498+250]
        val_accuracy[-1] = val_accuracy[-1][:4]
        val_accuracy[0] = val_accuracy[0][-5:]
        val_accuracy = [float(c.strip('[ ]')) for c in val_accuracy]
    return train_loss, val_accuracy


def set_architecture(arch:str, input_dim:int, seed: int):
    torch.manual_seed(seed)
    if arch=='S':
        return nn.Sequential(nn.Linear(input_dim,50),nn.Sigmoid(),nn.Linear(50,1))
    elif arch=='M':
        return nn.Sequential(nn.Linear(input_dim,20),nn.Sigmoid(),nn.Linear(20,20),
                             nn.Sigmoid(), nn.Linear(20,20), nn.Sigmoid(), nn.Linear(20,1))
    elif arch=='L':
        return nn.Sequential(nn.Linear(input_dim,50),nn.Sigmoid(),nn.Linear(50,50),
                             nn.Sigmoid(),nn.Linear(50,50),nn.Sigmoid(),nn.Linear(50,50),
                             nn.Sigmoid(),nn.Linear(50,50), nn.Sigmoid(), nn.Linear(50,1))
    elif arch == 'XL':
        return nn.Sequential(nn.Linear(input_dim, 50), nn.Sigmoid(), nn.Linear(50, 50),
                             nn.Sigmoid(), nn.Linear(50, 50), nn.Sigmoid(), nn.Linear(50, 50),
                             nn.Sigmoid(), nn.Linear(50, 50), nn.Sigmoid(),
                             nn.Linear(50, 50), nn.Sigmoid(), nn.Linear(50, 50), nn.Sigmoid(),
                             nn.Linear(50, 50), nn.Sigmoid(), nn.Linear(50, 50), nn.Sigmoid(),
                             nn.Linear(50, 50), nn.Sigmoid(), nn.Linear(50, 1))
    elif arch == 'XXL':
        return nn.Sequential(nn.Linear(input_dim, 50), nn.Sigmoid(), nn.Linear(50, 300),
                             nn.Sigmoid(), nn.Linear(300, 300), nn.Sigmoid(), nn.Linear(300, 300),
                             nn.Sigmoid(), nn.Linear(300, 300),nn.Sigmoid(), nn.Linear(300, 300),
                             nn.Sigmoid(), nn.Linear(300, 300),nn.Sigmoid(), nn.Linear(300, 300),
                             nn.Sigmoid(), nn.Linear(300, 300),nn.Sigmoid(), nn.Linear(300, 300),
                             nn.Sigmoid(), nn.Linear(300, 300),nn.Sigmoid(), nn.Linear(300, 300),
                             nn.Sigmoid(), nn.Linear(300, 50),nn.Linear(50, 1))
    elif arch == 'XXXL':
        return nn.Sequential(nn.Linear(input_dim, 50), nn.Sigmoid(), nn.Linear(50, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 50),nn.Linear(50, 1))

    elif arch == '4XL':
        return nn.Sequential(nn.Linear(input_dim, 50), nn.Sigmoid(), nn.Linear(50, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 500), nn.Sigmoid(), nn.Linear(500, 500),
                             nn.Sigmoid(), nn.Linear(500, 50),nn.Linear(50, 1))

    else:
        raise SystemError('Set an architecture in {S,M,L,XL,XXL,XXXL,4XL} and try again')