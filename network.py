import torch
import torchvision

def initialize_weights(model,seed):
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)


def get_pretrained_net(type_model, num_classes, seed, pretrained=False):
    torch.manual_seed(seed)
    if type_model=='resnet18':
        if pretrained:
            pretrainedmodel = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1, progress=True)
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            torch.nn.init.xavier_uniform(pretrainedmodel.fc.weight)
        else:
            pretrainedmodel = torchvision.models.resnet18()
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            initialize_weights(pretrainedmodel,seed)

    elif type_model=='resnet50':
        if pretrained:
            pretrainedmodel = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1, progress=True)
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            torch.nn.init.xavier_uniform(pretrainedmodel.fc.weight)
        else:
            pretrainedmodel = torchvision.models.resnet50()
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            initialize_weights(pretrainedmodel,seed)

    elif type_model=='resnet101':
        if pretrained:
            pretrainedmodel = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1, progress=True)
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            torch.nn.init.xavier_uniform(pretrainedmodel.fc.weight)
        else:
            pretrainedmodel = torchvision.models.resnet101()
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            initialize_weights(pretrainedmodel,seed)

    elif type_model=='resnet34':
        if pretrained:
            pretrainedmodel = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1, progress=True)
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            torch.nn.init.xavier_uniform(pretrainedmodel.fc.weight)
        else:
            pretrainedmodel = torchvision.models.resnet34()
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            initialize_weights(pretrainedmodel,seed)

    elif type_model=='resnet152':
        if pretrained:
            pretrainedmodel = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1, progress=True)
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            torch.nn.init.xavier_uniform(pretrainedmodel.fc.weight)
        else:
            pretrainedmodel = torchvision.models.resnet152()
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            initialize_weights(pretrainedmodel,seed)

    else:
        raise ValueError('Unrecognized network')

    return pretrainedmodel
    

