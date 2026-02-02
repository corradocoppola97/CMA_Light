import itertools
import torch
from main import train_model

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

def main():
    results = []
    for ds, arch in test_problems:
        for zeta, theta, delta, gamma in itertools.product(param_grid['zeta'], param_grid['theta'], param_grid['delta'], param_grid['gamma']):
            print(f"Training: ds={ds}, arch={arch}, zeta={zeta}, theta={theta}, delta={delta}, gamma={gamma}")
            try:
                history = train_model(
                    ds=ds,
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
                    verbose_train=False
                )
                val_loss = history['val_loss'][-1] if 'val_loss' in history and len(history['val_loss']) > 0 else None
                results.append({
                    'ds': ds,
                    'arch': arch,
                    'zeta': zeta,
                    'theta': theta,
                    'delta': delta,
                    'gamma': gamma,
                    'val_loss': val_loss
                })
            except Exception as e:
                print(f"Failed for ds={ds}, arch={arch}, zeta={zeta}, theta={theta}, delta={delta}, gamma={gamma}: {e}")
    # Sort and print best results
    results = [r for r in results if r['val_loss'] is not None]
    results.sort(key=lambda x: x['val_loss'])
    print("\nBest results:")
    for r in results[:10]:
        print(r)

if __name__ == "__main__":
    main()
