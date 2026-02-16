import itertools
import random
import time
import os
import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.optim.lbfgs
from fnn import FNN
from Dataset import create_dataset
from utils import *
from utils_cmalight import closure_reg


def train_model(x_train,
                x_test,
                y_train,
                y_test,
                arch: str,
                sm_root: str,
                opt: str,
                ep: int,
                time_limit: int,
                batch_size:int,
                seed: int,
                ID_history = '',
                verbose_train = False,
                shuffle = False,
                device = 'cpu',
                criterion = torch.nn.functional.mse_loss,
                ds_name: str = '',
                *args,
                **kwargs):

    if verbose_train: print('\n ------- Begin training process ------- \n')

    # Hardware
    if device is None: device = 'cuda' if (torch.cuda.is_available()) else 'cpu'
    device = torch.device(device)
    torch.cuda.empty_cache()

    # Setups
    history = {'train_loss': [], 'val_loss': [], 'time_4_epoch': [], 'step_size': [],
               'accepted': [], 'nfev': 0, 'Exit':[]}

    class _DatasetView:
        def __init__(self, x_train, x_test, y_train, y_test, csv_name=''):
            self.x_train = x_train
            self.x_test = x_test
            self.y_train = y_train
            self.y_test = y_test
            self.P = x_train.shape[0]
            self.P_test = x_test.shape[0]
            self.n = x_train.shape[1] if x_train.ndim > 1 else 1
            self.csv = csv_name
            self.idx = None

        def minibatch(self, first: int, last: int, test=False):
            if not test:
                if last > self.P:
                    last = self.P
                self.x_train_mb = self.x_train[first:last, :]
                self.y_train_mb = self.y_train[first:last].flatten()
            else:
                if last > self.P_test:
                    last = self.P_test
                self.x_test_mb = self.x_test[first:last, :]
                self.y_test_mb = self.y_test[first:last].flatten()

        def reshuffle(self, seed=100):
            idx = np.arange(self.P)
            np.random.seed(seed)
            np.random.shuffle(idx)
            self.x_train = self.x_train[idx]
            self.y_train = self.y_train[idx]

    dataset = _DatasetView(x_train, x_test, y_train, y_test, csv_name=ds_name)
    input_dim = dataset.n
    layers = set_architecture(arch,input_dim,seed)
    model = FNN(layers=layers).to(device)
    optimizer = set_optimizer(opt,model,*args,**kwargs)
    if opt == 'lbfgs' and batch_size != dataset.P:
        batch_size = dataset.P
        ep = 1
        if verbose_train==True: print(f'Setting batch size = {dataset.P}  and ep = 1 since you are using a full-batch method')

    if verbose_train: print("\n --------- Start Train --------- \n")

    # Train
    start_time_4_epoch = time.time()
    if opt == 'cma' or opt == 'nmcma' or opt=='cmal':
        fw0 = closure_reg(dataset,device,model,criterion)
        optimizer.set_fw0(fw0)
        history['nfev'] +=1
        f_before = fw0
        if opt=='cma':
            optimizer.set_reference(f_before=f_before)
        else:
            if opt=='cmal':
                optimizer.set_phi(fw0)
            else:
                optimizer.set_f_before(f_before=f_before)
    else:
        f_before = closure_reg(dataset,device,model,criterion)
        history['nfev'] += 1
    if opt == 'sgd':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)
    test_loss = closure_reg(dataset,device,model,criterion,test=True)
    history['train_loss'].append(f_before)
    history['val_loss'].append(test_loss)
    history['time_4_epoch'].append(0.0)
    if optimizer in {'nmcma','cma','ig','cmal'}:
        history['step_size'].append(optimizer.param_groups[0]['zeta'])

    # Training cycle - Epochs Level
    for epoch in range(ep):
        if verbose_train: print(f'Epoch n. {epoch+1}/{ep}')
        if time.time() - start_time_4_epoch >= time_limit:
            break
        if opt == 'cma' or opt == 'nmcma' or opt=='cmal':
            w_before = get_w(model)
            if opt == 'nmcma':
                M = optimizer.param_groups[0]['M']
                R_k = max(history['train_loss'][-M:])
                optimizer.set_Rk(R_k)

        # Training cycle - Single epoch level
        n_iterations = int(np.ceil(dataset.P/batch_size))
        f_tilde = 0
        for j in range(n_iterations):
            optimizer.zero_grad()
            dataset.minibatch(j * batch_size, (j + 1) * batch_size)
            x, y = dataset.x_train_mb.to(device), dataset.y_train_mb.to(device)
            y_pred = model(x)
            loss = criterion(target=y, input=y_pred)
            f_tilde += loss.item() * (len(x) / dataset.P)
            loss.backward()

            if opt=='lbfgs':
                optimizer.step(closure_reg,dataset=dataset,device=device,mod=model,loss_fun=criterion)
            else:
                optimizer.step()
            if opt == 'sgd':
                scheduler.step()

        # CMA support functions
        if opt == 'cma':
            model, history, f_before, f_after, exit_type = optimizer.control_step(model,w_before,closure_reg,
                                                    dataset,device,criterion,history,epoch)
            optimizer.set_reference(f_before=f_after)

            elapsed_time_4_epoch = time.time() - start_time_4_epoch

            if history['step_size'][-1] <= 1e-15:
                history['comments'] = f'Forced stopping at epoch {epoch}'
                break

        #NMCMA support functions
        elif opt=='nmcma':
            model, history, f_before, f_after, exit_type = optimizer.control_step(model, w_before, closure_reg,
                                                                                  dataset, device, criterion, history,
                                                                                  epoch)
            optimizer.set_f_before(f_before=f_after)

            elapsed_time_4_epoch = time.time() - start_time_4_epoch

            if history['step_size'][-1] <= 1e-15:
                history['comments'] = f'Forced stopping at epoch {epoch}'
                break

        #CMAL support functions
        elif opt=='cmal':
            optimizer.set_f_tilde(f_tilde)
            phi = optimizer.phi
            model, history, f_after, exit = optimizer.control_step(model, w_before, closure_reg,
                                                                   dataset, device, criterion, history, epoch)
            optimizer.set_phi(min(f_tilde, f_after, phi))
            elapsed_time_4_epoch = time.time() - start_time_4_epoch

            if history['step_size'][-1] <= 1e-15:
                history['comments'] = f'Forced stopping at epoch {epoch}'
                break


        else: # Compute the training loss after if you are not using CMA/NMCMA
            elapsed_time_4_epoch = time.time() - start_time_4_epoch
            f_after = closure_reg(dataset,device,model,criterion) #The cpu time for this operation is excluded because you don't need it



        # Test
        test_loss = closure_reg(dataset,device,model,criterion,test = True)

        # Update history
        try:
            history['train_loss'].append(f_after.item())
        except:
            history['train_loss'].append(f_after)
        try:
            history['val_loss'].append(test_loss.item())
        except:
            history['val_loss'].append(test_loss)
        history['time_4_epoch'].append(elapsed_time_4_epoch)

        if opt == 'ig':
            history['step_size'].append(optimizer.param_groups[0]['zeta'])
            optimizer.update_zeta()
            if history['step_size'][-1] <= 1e-15:
                history['comments'] = f'Forced stopping at epoch {epoch}'
                break
        if verbose_train: print(f'End Epoch {epoch}   Train Loss:{f_after:3e}  Elapsed time:{elapsed_time_4_epoch:3f} \n ')


        # Empty CUDA cache
        torch.cuda.empty_cache()

        #If needed, reshuffle
        if shuffle == True:
            dataset.reshuffle(seed=random.randint(1,1000))

    # Operations after training
    ds_base = os.path.splitext(dataset.csv)[0] if dataset.csv else 'dataset'
    torch.save(history,sm_root + 'history_'+opt+'_'+arch+'_'+ds_base+'_'+ID_history+'.txt')
    if verbose_train: print('\n - Finished Training - \n')
    return history

# Define the grid of hyperparameters
param_grid = {
    'zeta': [0.01, 0.05, 0.1],
    'theta': [0.5, 0.75, 0.9],
    'delta': [0.8, 0.9, 1.0],
    'gamma': [1e-6, 1e-5, 1e-4]
}

# List of (ds, arch) problems to test
# Expanded to around ten diverse pairs
# You can adjust these as needed for your datasets and architectures
test_problems = [
    ('Mv', 'S'),
    ('Mv', 'L'),
    ('California', 'M'),
    ('California', 'XL'),
    ('Protein', 'S'),
    ('Protein', 'L'),
    ('Ailerons', 'M'),
    ('Ailerons', 'XL'),
    ('BlogFeedback', 'S'),
    ('BlogFeedback', 'L'),
    ('Covtype', 'M'),
    ('Covtype', 'XL'),
]

# Other fixed parameters
sm_root = 'grid_search_results/'
ep = 250  # Fewer epochs for grid search speed
time_limit = 120  # seconds
batch_size = 128
seed = 1
opt = 'cmal'  # Coff_Ig optimizer

def main(k_folds: int = 5):
    results = []
    for ds, arch in test_problems:
        print(f"\nDataset: {ds}  Arch: {arch}  -- KFold={k_folds}")
        try:
            ds_obj = create_dataset(ds, frac=0.75)
        except Exception as e:
            print(f"Could not load dataset {ds}: {e}")
            continue

        X = ds_obj.x
        Y = ds_obj.y
        n_samples = X.shape[0]
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

        for zeta, theta, delta, gamma in itertools.product(param_grid['zeta'], param_grid['theta'], param_grid['delta'], param_grid['gamma']):
            combo_desc = f"ds={ds}, arch={arch}, zeta={zeta}, theta={theta}, delta={delta}, gamma={gamma}"
            print(f"Training: {combo_desc}")
            fold_val_losses = []
            failed = False

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(n_samples))):
                try:
                    train_idx_t = torch.from_numpy(train_idx).long()
                    val_idx_t = torch.from_numpy(val_idx).long()
                    x_tr = X[train_idx_t]
                    y_tr = Y[train_idx_t]
                    x_val = X[val_idx_t]
                    y_val = Y[val_idx_t]

                    history = train_model(
                        x_train=x_tr,
                        x_test=x_val,
                        y_train=y_tr,
                        y_test=y_val,
                        arch=arch,
                        sm_root=sm_root,
                        opt=opt,
                        ep=ep,
                        time_limit=time_limit,
                        batch_size=batch_size,
                        seed=seed,
                        zeta=zeta,
                        theta=theta,
                        delta=delta,
                        gamma=gamma,
                        verbose_train=False,
                        ds_name=f"{ds}_fold{fold_idx}",
                        ID_history=f"seed_{seed}_fold_{fold_idx}"
                    )

                    val_loss = history['val_loss'][-1] if 'val_loss' in history and len(history['val_loss']) > 0 else None
                    if val_loss is None:
                        raise RuntimeError('No validation loss returned')
                    fold_val_losses.append(float(val_loss))

                except Exception as e:
                    print(f"  Fold {fold_idx} failed: {e}")
                    failed = True
                    break

            mean_val = float(np.mean(fold_val_losses)) if (len(fold_val_losses) > 0 and not failed) else None
            results.append({
                'ds': ds,
                'arch': arch,
                'zeta': zeta,
                'theta': theta,
                'delta': delta,
                'gamma': gamma,
                'mean_val_loss': mean_val
            })

    # Filter and print best results
    valid_results = [r for r in results if r['mean_val_loss'] is not None]
    valid_results.sort(key=lambda x: x['mean_val_loss'])
    print("\nBest results:")
    for r in valid_results[:10]:
        print(r)


if __name__ == "__main__":
    main()
