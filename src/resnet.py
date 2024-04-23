import torchvision
from torchvision import models
from blitz.modules import BayesianLinear, BayesianConv2d
import torch.nn as nn

def build_resnet_model(pretrained=True, fine_tune=True, bayes_type=True, num_classes=2):
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
        model.conv1 = BayesianConv2d(
                        in_channels=model.conv1.in_channels,
                        out_channels=model.conv1.out_channels,
                        kernel_size=model.conv1.kernel_size,
                        stride=model.conv1.stride,
                        # bias=model.conv1.bias
                    )
        for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for bottleneck in layer:
                bottleneck.conv1 = BayesianConv2d(
                    in_channels=bottleneck.conv1.in_channels,
                    out_channels=bottleneck.conv1.out_channels,
                    kernel_size=bottleneck.conv1.kernel_size,
                    stride=bottleneck.conv1.stride,
                    # bias=bottleneck.conv1.bias
                )
                bottleneck.conv2 = BayesianConv2d(
                    in_channels=bottleneck.conv2.in_channels,
                    out_channels=bottleneck.conv2.out_channels,
                    kernel_size=bottleneck.conv2.kernel_size,
                    stride=bottleneck.conv2.stride,
                    # bias=bottleneck.conv2.bias
                )
                bottleneck.conv3 = BayesianConv2d(
                    in_channels=bottleneck.conv3.in_channels,
                    out_channels=bottleneck.conv3.out_channels,
                    kernel_size=bottleneck.conv3.kernel_size,
                    stride=bottleneck.conv3.stride,
                    # bias=bottleneck.conv3.bias
                )

        # Classifier
        block = model.fc
        model.fc = BayesianLinear(in_features=block.in_features, out_features=num_classes)

    else:
        # non Bayesian
        block = model.fc
        model.fc = nn.Linear(in_features=block.in_features, out_features=num_classes)

    return model
