import torchvision
from torchvision import models
from blitz.modules import BayesianLinear, BayesianConv2d
import torch.nn as nn

def replace_conv_with_bayesian(conv):
    return BayesianConv2d(in_channels=conv.in_channels,
                          out_channels=conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups)

def convert_bottleneck_to_bayesian(bottleneck):
    bottleneck.conv1 = replace_conv_with_bayesian(bottleneck.conv1)
    bottleneck.conv2 = replace_conv_with_bayesian(bottleneck.conv2)
    bottleneck.conv3 = replace_conv_with_bayesian(bottleneck.conv3)
    if bottleneck.downsample is not None:
        for i, layer in enumerate(bottleneck.downsample):
            if isinstance(layer, nn.Conv2d):
                bottleneck.downsample[i] = replace_conv_with_bayesian(layer)


def build_resnet_model(pretrained=True, fine_tune=True, bayes_type='all', num_classes=2):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
    if pretrained:
        model = torchvision.models.resnet50(weights='DEFAULT')
    else:
        model = torchvision.models.resnet50(weights=None)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    if bayes_type.lower() == 'last':
        # Change the final classification head.
        block = model.fc
        model.fc = BayesianLinear(in_features=block.in_features, out_features=num_classes)
    elif bayes_type.lower() == 'all':
        # Replace the first conv layer
        model.conv1 = replace_conv_with_bayesian(model.conv1)
        # Convert all bottleneck blocks
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(model, layer_name)
            for bottleneck in layer:
                convert_bottleneck_to_bayesian(bottleneck)

        # Replace the final classifier layer
        block = model.fc
        model.fc = BayesianLinear(in_features=block.in_features, out_features=num_classes)

    else:
        # non Bayesian
        block = model.fc
        model.fc = nn.Linear(in_features=block.in_features, out_features=num_classes)

    return model
