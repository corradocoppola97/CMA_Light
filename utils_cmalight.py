import torch
from tqdm import tqdm



def closure_reg(dataset,
            device: torch.device,
            mod,
            loss_fun=torch.nn.CrossEntropyLoss(),
            batch_size: int = 128,
            test: bool = False):

    if test:
        x_all, y_all, P = dataset.x_test, dataset.y_test, dataset.P_test
    else:
        x_all, y_all, P = dataset.x_train, dataset.y_train, dataset.P

    loss = 0.0

    with torch.no_grad():
        with tqdm(range(0, P, batch_size), unit="step", position=0, leave=False) as tepoch:
            for i in tepoch:
                tepoch.set_description("Validation" if test else "Training")

                # use minibatch method to extract slice
                dataset.minibatch(i, i + batch_size, test=test)
                if test:
                    x = dataset.x_test_mb.to(device)
                    y = dataset.y_test_mb.to(device)
                else:
                    x = dataset.x_train_mb.to(device)
                    y = dataset.y_train_mb.to(device)

                batch_loss = loss_fun(mod(x), y.long()).item()
                loss += (len(x) / P) * batch_loss

    return loss



