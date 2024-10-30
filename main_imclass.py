import os
import time
from utils import closure, count_parameters, set_optimizer, accuracy
from cmalight import get_w
from network import get_pretrained_net
import torch
import torchvision
from torch.utils.data import Subset
from warnings import filterwarnings
from tqdm import tqdm
from cma import CMA

filterwarnings('ignore')


def train_model(sm_root: str,
                opt: str,
                ep: int,
                ds: str,
                net_name: str,
                n_class: int,
                history_ID: str,
                seed: int,
                dts_train,
                dts_test,
                modelpth = None,
                savemodel = False) -> dict:
    print('\n ------- Begin training process ------- \n')

    # Hardware
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    # Model
    model = get_pretrained_net(net_name, num_classes=n_class,pretrained=True,seed=seed).to(device)
    if modelpth is not None:
        model = torch.load(modelpth,map_location=torch.device('cpu'))
    print('\n The model has: {} trainable parameters'.format(count_parameters(model)))
    # Loss
    criterion = torch.nn.CrossEntropyLoss()
    # Optimizer
    optimizer = set_optimizer(opt, model)
    # Initial Setup
    min_acc = 0
    t1 = time.time()
    fw0 = closure(dts_train, model, criterion, device)
    t2 = time.time()
    time_compute_fw0 = t2 - t1  # To be added to the elapsed time in case we are using CMA Light (information used)
    initial_val_loss = closure(dts_test, model, criterion, device)
    train_accuracy = accuracy(dts_train, model, device)
    val_acc = accuracy(dts_test, model, device)
    f_tilde = fw0

    if opt == 'cmal':
        optimizer.set_f_tilde(f_tilde)
        optimizer.set_phi(f_tilde)
        optimizer.set_fw0(fw0)
    if opt == 'cma':
        optimizer.set_fw0(fw0)
        optimizer.set_reference(fw0)


    history = {'train_loss': [fw0], 'val_loss': [initial_val_loss], 'train_acc': [train_accuracy],
               'val_accuracy': [val_acc], 'step_size': [],
               'time_4_epoch': [0.0], 'nfev': 1, 'accepted': [], 'Exit': [], 'comments': [],
               'elapsed_time': [0.0], 'f_tilde': []}

    # Train
    for epoch in range(ep):
        start_time = time.time()
        model.train()
        f_tilde = 0
        if opt == 'cmal' or opt == 'cma':
            w_before = get_w(model)
        with tqdm(dts_train, unit="step", position=0, leave=True) as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{ep} - Training")
                x, y = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                f_tilde += loss.item() * (len(x) / 1024)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(f_tilde=f_tilde)

        history['f_tilde'].append(f_tilde)

        # CMAL support functions
        if opt == 'cmal':
            optimizer.set_f_tilde(f_tilde)
            model, history, f_after, exit = optimizer.control_step(model, w_before, closure, dts_train, device,
                                                                   criterion, history, epoch)
            optimizer.set_phi(min(f_tilde, f_after, optimizer.phi))
        else:
            if opt == 'cma':
                model, history, f_before, f_after, exit_type = optimizer.control_step(model, w_before, closure,
                                                                                      dts_train, device, criterion,
                                                                                      history, epoch)
                optimizer.set_reference(f_before=f_after)

            else:
                pass

        elapsed_time_noVAL = time.time() - start_time

        # Validation
        model.eval()
        val_loss = closure(dts_test, model, criterion, device)
        val_acc = accuracy(dts_test, model, device)
        elapsed_time_4_epoch = time.time() - start_time
        train_accuracy = accuracy(dts_train, model, device)
        real_train_loss = closure(dts_train,model,criterion,device)
        history['time_4_epoch'].append(history['time_4_epoch'][-1] + elapsed_time_noVAL)
        print(f'Time: {history["elapsed_time"][-1] + elapsed_time_noVAL}')
        print(f'train_loss = {real_train_loss}   train_acc = {train_accuracy}')
        print(f'val_loss = {val_loss}   val_acc = {val_acc}')
        history['train_loss'].append(real_train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_accuracy'].append(val_acc)
        history['elapsed_time'].append(history['elapsed_time'][-1] + elapsed_time_4_epoch)
        if epoch == 0 and opt == 'cmal':
            history['time_4_epoch'][-1] += time_compute_fw0
            history['elapsed_time'][-1] += time_compute_fw0
        # Save data during training
        if min_acc < val_acc and savemodel == True:
            torch.save(model, sm_root + 'train_' + opt + '_' + net_name + '_' + ds + '_model_best'+str(seed)+'.pth')
            min_acc = val_acc
            print('\n - New best Val-ACC: {:.3f} at epoch {} - \n'.format(min_acc, ep + 1))

        torch.save(history, sm_root + 'history_' + opt + '_' + net_name + '_' + ds + '_' + history_ID +'.txt')
    print('\n - Finished Training - \n')
    torch.save(history, sm_root + 'history_' + opt + '_' + net_name + '_' + ds + '_' + history_ID + '.txt')
    return history

seed = 1
torch.manual_seed(seed)
transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                            torchvision.transforms.RandomRotation(10),
                                            torchvision.transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                                                               saturation=0.2),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='\data', train=True, download=True, transform=transform)
#trainset = Subset(trainset,range(1024))
testset = torchvision.datasets.CIFAR10(root='\data', train=False, download=True, transform=transform)
#testset = Subset(testset,range(256))
num_classes = 10
bs = 128
#try:
#    os.mkdir('prove_imclass_28apr')
#except:
#    os.chdir('prove_imclass_28apr')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=False)  # Togliere random reshuffle --> shuffle=False
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)
tl = [(x,y) for x,y in trainloader]
tstl = [(x,y) for x,y in testloader]
for rete in ['resnet18']:#,'resnet34','resnet50','resnet101','resnet150']:
    for algo in ['cma','cmal','adam','adagrad','adadelta']:
        history = train_model(sm_root='',
                          opt=algo,
                          ep=1,
                          ds='cifar10',
                          net_name=rete,
                          n_class=10,
                          history_ID='seed_'+str(seed),
                          dts_train=tl,
                          dts_test=tstl,
                          seed=seed,
                          savemodel=False)

