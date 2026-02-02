import time
from warnings import filterwarnings

import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import closure, count_parameters, set_optimizer, accuracy
from coff_ig import get_w
from network import get_pretrained_net

filterwarnings('ignore')


def train_model(
    sm_root: str,
    opt: str,
    ep: int,
    ds: str,
    net_name: str,
    n_class: int,
    history_ID: str,
    seed: int,
    dts_train,
    dts_test,
    modelpth=None,
    savemodel: bool = False,
) -> dict:
    print('\n ------- Begin training process ------- \n')

    # Hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    # Model
    model = get_pretrained_net(net_name, num_classes=n_class, pretrained=True, seed=seed).to(device)
    if modelpth is not None:
        model = torch.load(modelpth, map_location=device).to(device)

    print(f'\n The model has: {count_parameters(model)} trainable parameters')

    # Loss + Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = set_optimizer(opt, model)

    # Baselines
    t1 = time.time()
    with torch.inference_mode():
        fw0 = closure(dts_train, model, criterion, device)
    t2 = time.time()
    time_compute_fw0 = t2 - t1
    with torch.inference_mode():
        initial_val_loss = closure(dts_test, model, criterion, device)
        train_accuracy = accuracy(dts_train, model, device)
        val_acc = accuracy(dts_test, model, device)

    f_tilde = fw0

    if opt == 'cmal':
        optimizer.set_f_tilde(f_tilde)
        optimizer.set_phi(f_tilde)
        optimizer.set_fw0(fw0)
    elif opt == 'cma':
        optimizer.set_fw0(fw0)
        optimizer.set_reference(fw0)

    history = {
        'train_loss': [fw0],
        'val_loss': [initial_val_loss],
        'train_acc': [train_accuracy],
        'val_accuracy': [val_acc],
        'step_size': [],
        'time_4_epoch': [0.0],
        'nfev': 1,
        'accepted': [],
        'Exit': [],
        'comments': [],
        'elapsed_time': [0.0],
        'f_tilde': [],
    }

    # Training loop
    for epoch in range(ep):
        start_time = time.time()
        model.train()
        f_tilde = 0.0

        if opt in ('cmal', 'cma'):
            w_before = get_w(model)

        with tqdm(dts_train, unit='step', position=0, leave=True) as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{ep} - Training")
            for x, y in tepoch:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                y_pred = model(x)
                loss = criterion(y_pred, y)

                f_tilde += loss.item() * (len(x) / 1024)

                loss.backward()
                optimizer.step()

                tepoch.set_postfix(f_tilde=f"{f_tilde:.4f}")

        history['f_tilde'].append(f_tilde)

        # Control step
        if opt == 'cmal':
            optimizer.set_f_tilde(f_tilde)
            model, history, f_after, exit_flag = optimizer.control_step(
                model, w_before, closure, dts_train, device, criterion, history, epoch
            )
            optimizer.set_phi(min(f_tilde, f_after, optimizer.phi))
        elif opt == 'cma':
            model, history, f_before, f_after, exit_type = optimizer.control_step(
                model, w_before, closure, dts_train, device, criterion, history, epoch
            )
            optimizer.set_reference(f_before=f_after)

        elapsed_time_noVAL = time.time() - start_time

        # Validation
        model.eval()
        with torch.inference_mode():
            val_loss = closure(dts_test, model, criterion, device)
            val_acc = accuracy(dts_test, model, device)
            train_accuracy = accuracy(dts_train, model, device)
            real_train_loss = closure(dts_train, model, criterion, device)

        elapsed_time_4_epoch = time.time() - start_time

        # Time bookkeeping
        history['time_4_epoch'].append(history['time_4_epoch'][-1] + elapsed_time_noVAL)
        history['elapsed_time'].append(history['elapsed_time'][-1] + elapsed_time_4_epoch)

        if epoch == 0 and opt == 'cmal':
            history['time_4_epoch'][-1] += time_compute_fw0
            history['elapsed_time'][-1] += time_compute_fw0

        # Logging
        print(f'Time: {history["elapsed_time"][-1]:.2f}s '
              f'| train_loss={real_train_loss:.4f} train_acc={train_accuracy:.4f} '
              f'| val_loss={val_loss:.4f} val_acc={val_acc:.4f}')

        # History metrics
        history['train_loss'].append(real_train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_accuracy'].append(val_acc)
        history['step_size'].append(optimizer.param_groups[0].get('zeta', None))

        # Save best model
        if savemodel and val_acc > max(history['val_accuracy'][:-1]):
            torch.save(model, f'{sm_root}train_{opt}_{net_name}_{ds}_model_best{seed}.pth')
            print(f'\n - New best Val-ACC: {val_acc:.3f} at epoch {epoch + 1} - \n')

        # Save history
        torch.save(history, f'{sm_root}history_{opt}_{net_name}_{ds}_{history_ID}.txt')

    print('\n - Finished Training - \n')
    torch.save(history, f'{sm_root}history_{opt}_{net_name}_{ds}_{history_ID}.txt')
    return history


if __name__ == '__main__':
    seed = 1
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Transforms
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(10),
        torchvision.transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Choose dataset (switch here)
    DATASET = "CIFAR100"

    if DATASET == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        n_class = 10
    else:
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        n_class = 100

    # DataLoaders
    pin_mem = torch.cuda.is_available()
    num_workers = 2  # stable across OSes
    trainloader = DataLoader(
        trainset, batch_size=64, shuffle=True,
        pin_memory=pin_mem, num_workers=num_workers, persistent_workers=(num_workers > 0)
    )
    testloader = DataLoader(
        testset, batch_size=64, shuffle=False,
        pin_memory=pin_mem, num_workers=num_workers, persistent_workers=(num_workers > 0)
    )

    for rete in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        for algo in ['sgd', 'cmal', 'adam','adagrad','adadelta']:
            try:
                train_model(
                    sm_root='',
                    opt=algo,
                    ep=50,
                    ds=DATASET.lower(),
                    net_name=rete,
                    n_class=n_class,
                    history_ID=f'seed_{seed}',
                    dts_train=trainloader,
                    dts_test=testloader,
                    seed=seed,
                    savemodel=False,
                )
            except Exception as e:
                print(f'[SKIP] net={rete}, opt={algo} due to error: {e}')
