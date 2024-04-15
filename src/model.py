import torchvision
from blitz.modules import BayesianLinear
import torch.nn as nn

# https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/
def build_effnet_model(pretrained=True, fine_tune=True, bayes_last=True, num_classes=2):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
    if pretrained:
        model = torchvision.models.efficientnet_b0(weights='DEFAULT')
    else:
        model = torchvision.models.efficientnet_b0(weights=None)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    if bayes_last:
        # Change the final classification head.
        block = model.classifier[1]
        model.classifier[1] = BayesianLinear(in_features=block.in_features, out_features=num_classes)
    elif not bayes_last:
        block = model.classifier[1]
        model.classifier[1] = nn.Linear(in_features=block.in_features, out_features=num_classes)

    return model
