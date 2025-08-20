import torch
from tqdm import tqdm

def closure(data_loader: torch.utils.data.DataLoader,
            device: torch.device,
            mod,
            loss_fun = torch.nn.CrossEntropyLoss()): #mettere in utils
    loss = 0
    P = len(data_loader)
    with torch.no_grad():
        with tqdm(data_loader, unit="step", position=0, leave=True) as tepoch:
            for batch in tepoch:
                tepoch.set_description("Validation")
                x, y = batch[0].to(device), batch[1].to(device)
                batch_loss = loss_fun(input=mod(x), target=y).item()
                loss += (len(x) / P) * batch_loss
    return loss



