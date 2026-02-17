import time
import csv
import itertools
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from sklearn.model_selection import KFold
import numpy as np


def get_resnet18(num_classes: int = 10) -> nn.Module:
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_epoch(model, device, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(y).sum().item()
        total += x.size(0)

    return running_loss / total, correct / total


def evaluate(model, device, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            running_loss += loss.item() * x.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(y).sum().item()
            total += x.size(0)

    return running_loss / total, correct / total


def make_param_grid(grid: Dict[str, list]):
    keys = list(grid.keys())
    for vals in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, vals))


def grid_search_kfold(
    optimizer_name: str,
    param_grid: Dict[str, list],
    k: int = 5,
    epochs: int = 10,
    batch_size: int = 128,
    device: torch.device = None,
):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Using device: {device}')

    # Dataset & transforms (standard CIFAR-10 preprocessing)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    # For validation we will use the same dataset split via indices; transformations above suffice.

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    results = []

    for params in make_param_grid(param_grid):
        print('\nTesting params:', params)
        fold_scores = []
        fold = 0
        for train_idx, val_idx in kf.split(np.arange(len(cifar_train))):
            fold += 1
            # Create subsets with appropriate transforms
            train_subset = Subset(cifar_train, train_idx)
            val_subset = Subset(torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_val), val_idx)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

            # fresh model per fold
            model = get_resnet18(num_classes=10).to(device)
            criterion = nn.CrossEntropyLoss()

            # build optimizer
            if optimizer_name.lower() == 'adam':
                lr = params.get('lr', 1e-3)
                beta1 = params.get('beta1', 0.9)
                weight_decay = params.get('weight_decay', 0.0)
                optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=weight_decay)
            elif optimizer_name.lower() == 'adagrad':
                lr = params.get('lr', 1e-2)
                lr_decay = params.get('lr_decay', 0.0)
                weight_decay = params.get('weight_decay', 0.0)
                optimizer = optim.Adagrad(model.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=weight_decay)
            else:
                raise ValueError('Unsupported optimizer: ' + optimizer_name)

            # train for fixed number of epochs
            best_val_acc = 0.0
            for epoch in range(epochs):
                t0 = time.time()
                train_loss, train_acc = train_epoch(model, device, train_loader, criterion, optimizer)
                val_loss, val_acc = evaluate(model, device, val_loader, criterion)
                t1 = time.time()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                print(f'Fold {fold} Epoch {epoch+1}/{epochs} - train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, time: {t1-t0:.1f}s')

            fold_scores.append(best_val_acc)

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))
        print(f'Params {params} -> mean val acc: {mean_score:.4f} (+/- {std_score:.4f})')
        result = {'optimizer': optimizer_name, **params, 'mean_val_acc': mean_score, 'std_val_acc': std_score}
        results.append(result)

    # return results sorted by mean_val_acc desc
    results_sorted = sorted(results, key=lambda r: r['mean_val_acc'], reverse=True)
    return results_sorted


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Select hyperparameter grids (reasonable intervals)
    adam_grid = {
        'lr': [1e-4, 5e-4, 1e-3, 5e-3],
        'beta1': [0.9, 0.95],
        'weight_decay': [0.0, 1e-4, 1e-3],
    }

    adagrad_grid = {
        'lr': [1e-3, 5e-3, 1e-2, 5e-2],
        'lr_decay': [0.0, 1e-2],
        'weight_decay': [0.0, 1e-4],
    }

    k_folds = 5
    epochs = 10
    batch_size = 128

    print('Starting grid search for Adam...')
    adam_results = grid_search_kfold('adam', adam_grid, k=k_folds, epochs=epochs, batch_size=batch_size, device=device)
    print('\nTop 5 Adam results:')
    for r in adam_results[:5]:
        print(r)

    print('\nStarting grid search for Adagrad...')
    adagrad_results = grid_search_kfold('adagrad', adagrad_grid, k=k_folds, epochs=epochs, batch_size=batch_size, device=device)
    print('\nTop 5 Adagrad results:')
    for r in adagrad_results[:5]:
        print(r)

    # Save to CSV
    out_file = 'gridsearch_results.csv'
    keys = set()
    for r in adam_results + adagrad_results:
        keys.update(r.keys())
    keys = sorted(list(keys))
    with open(out_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in adam_results + adagrad_results:
            writer.writerow(r)

    print(f'Grid search finished. Results saved to {out_file}')


if __name__ == '__main__':
    main()
