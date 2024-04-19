import torchvision
from torchvision import models
from blitz.modules import BayesianLinear, BayesianConv2d
import torch.nn as nn

# https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/
def build_effnet_model(pretrained=True, fine_tune=True, bayes_type=True, num_classes=2):
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
    if bayes_type.lower() == 'last':
        # Change the final classification head.
        block = model.classifier[1]
        model.classifier[1] = BayesianLinear(in_features=block.in_features, out_features=num_classes)
    elif bayes_type.lower() == 'all':
        # First and last feature are built differnt than features 1 - 7
        # Feature 0
        block = model.features[0]
        if isinstance(block, torchvision.ops.misc.Conv2dNormActivation):
            block[0] = BayesianConv2d(
                in_channels=block[0].in_channels,
                out_channels=block[0].out_channels,
                kernel_size=block[0].kernel_size,
                stride=block[0].stride,
                padding=block[0].padding,
                bias=block[0].bias
            )

        # Feature 1 - 7
        # Following the nested for loops we go from:
        # features -> MBConv
        # MBConv -> block
        # block -> each layer, where block[0] is our Conv2d layer and we replace it with BayesianConv2d
        for name, module in enumerate(model.features[1:8]):
            numMBVConv = len(module)
            for i in range(numMBVConv):
                mbconv_module = module[i]
                for idx, block in enumerate(mbconv_module.block):
                    if isinstance(block, torchvision.ops.misc.Conv2dNormActivation):
                        block[0] = BayesianConv2d(
                            in_channels=block[0].in_channels,
                            out_channels=block[0].out_channels,
                            kernel_size=block[0].kernel_size,
                            groups=block[0].groups,
                            stride=block[0].stride,
                            padding=block[0].padding,
                            bias=block[0].bias
                        )

        # Feature 8
        block = model.features[8]
        if isinstance(block, torchvision.ops.misc.Conv2dNormActivation):
            block[0] = BayesianConv2d(
                in_channels=block[0].in_channels,
                out_channels=block[0].out_channels,
                kernel_size=block[0].kernel_size,
                stride=block[0].stride,
                bias=block[0].bias
            )

        # Classifier
        block = model.classifier[1]
        model.classifier[1] = BayesianLinear(block.in_features, num_classes)

    else:
        block = model.classifier[1]
        model.classifier[1] = nn.Linear(in_features=block.in_features, out_features=num_classes)

    return model
