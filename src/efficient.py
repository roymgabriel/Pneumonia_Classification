import torch
from torch import nn
from torchvision import models


# Base class for each EfficientNet scale. Do not create instances of this base class, use EfficientNetB# classes below.
class EfficientNetBase(nn.Module):
    def __init__(self, cfg, efficientnet_initializer):
        super().__init__()
        # Load the EfficientNet model. If string indicating which pretrained weights to use are provided in cfg.MODEL_WEIGHTS, use them.
        self.efficientnet = efficientnet_initializer(
            weights=cfg.MODEL_WEIGHTS) if cfg.MODEL_WEIGHTS else efficientnet_initializer()
        # Modify the output layer to match the number of classes in your task
        in_features = self.efficientnet.classifier[1].in_features
        # Replace the output layer with a new fully connected layer with number of outputs for our task
        self.efficientnet.classifier[1] = nn.Linear(in_features, cfg.NUM_CLASSES)

    def forward(self, x):
        x = self.efficientnet(x)
        return x

    # Perform Monte Carlo Dropout with this mimic forward method.
    # Splits model into its constituent blocks, and performs dropouts in between each block. Does not use preexisting dropouts.
    # The dropout probability list is in order of when it appears in the model. This only includes dropouts added in mcd_forward.
    # This should be called within a torch.no_grad() block while model is in evaluation mode with model.eval().
    # The way this is formatted needs to follow the forward() method of the original model, just running (x) through the model's
    # features and classifier is not enough!! There are parts missing from the model summary that are handled in the model's forward method that
    # mcd_forward also needs to do to mimic the model's forward pass. The forward method can be viewed in the source code for the pytorch model,
    # it will most likely be in the base class definition that encompasses multiple versions of the model, not the specific model.
    # For EfficientNet, the forward method was under the EfficientNet base class, not models.efficientnet_b0.
    def mcd_forward(self, x, dropout_prob_list):
        # EfficientNet is already structured as a Sequential of Sequential blocks, so can simply index each block.
        x = self.efficientnet.features[0](x)
        # Perform dropout on x with the first probability in dropout_prob_list.
        x = nn.functional.dropout(x, dropout_prob_list[0])
        # Index the second block of model and perform inference on that block.
        x = self.efficientnet.features[1](x)
        # Perform dropout on x with the second probability in dropout_prob_list.
        x = nn.functional.dropout(x, dropout_prob_list[1])
        x = self.efficientnet.features[2](x)  # Third block.
        x = nn.functional.dropout(x, dropout_prob_list[2])
        x = self.efficientnet.features[3](x)  # Fourth block.
        x = nn.functional.dropout(x, dropout_prob_list[3])
        x = self.efficientnet.features[4](x)  # Fifth block.
        x = nn.functional.dropout(x, dropout_prob_list[4])
        x = self.efficientnet.features[5](x)  # Sixth block.
        x = nn.functional.dropout(x, dropout_prob_list[5])
        x = self.efficientnet.features[6](x)  # Seventh block.
        x = nn.functional.dropout(x, dropout_prob_list[6])
        x = self.efficientnet.features[7](x)  # Eighth block.
        x = nn.functional.dropout(x, dropout_prob_list[7])
        x = self.efficientnet.features[8](x)  # Ninth block.
        x = nn.functional.dropout(x, dropout_prob_list[8])

        # Perform the average pooling and flatten. This comes from the original model's forward method before the classifier.
        x = self.efficientnet.avgpool(x)
        x = torch.flatten(x, 1)

        # Perform dropout here, as the model has a dropout here so it is probably a good spot to put one.
        x = nn.functional.dropout(x, dropout_prob_list[9])
        # Index into final linear layer of classifier, skipping the model's dropout layer at index 0 of the classifier.
        x = self.efficientnet.classifier[1](x)

        return x


# The B# in the below classes indicates the size of the model, with B0 being the smallest model and B7 the largest.
# These just pass the pytorch function for initializing their respective architecture to the EfficientNetBase init, which runs it.
class EfficientNetB0(EfficientNetBase):
    def __init__(self, cfg):
        super().__init__(cfg, models.efficientnet_b0)


class EfficientNetB1(EfficientNetBase):
    def __init__(self, cfg):
        super().__init__(cfg, models.efficientnet_b1)


class EfficientNetB2(EfficientNetBase):
    def __init__(self, cfg):
        super().__init__(cfg, models.efficientnet_b2)


class EfficientNetB3(EfficientNetBase):
    def __init__(self, cfg):
        super().__init__(cfg, models.efficientnet_b3)


class EfficientNetB4(EfficientNetBase):
    def __init__(self, cfg):
        super().__init__(cfg, models.efficientnet_b4)


class EfficientNetB5(EfficientNetBase):
    def __init__(self, cfg):
        super().__init__(cfg, models.efficientnet_b5)


class EfficientNetB6(EfficientNetBase):
    def __init__(self, cfg):
        super().__init__(cfg, models.efficientnet_b6)


class EfficientNetB7(EfficientNetBase):
    def __init__(self, cfg):
        super().__init__(cfg, models.efficientnet_b7)


# Base class for the efficientnets that do transfer learning. Subclass of EfficientNetBase. Again, do not use this directly, use classes below.
class EfficientNetTransfer(EfficientNetBase):
    def __init__(self, cfg, efficientnet_initializer):
        super().__init__(cfg, efficientnet_initializer)
        # Freeze all parameters, except last fc layer
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        # Set the classifier layers to be unfrozen
        for param in self.efficientnet.classifier.parameters():
            param.requires_grad = True


# The B# in the below classes indicates the size of the model, with B0 being the smallest model and B7 the largest.
# These just pass the pytorch function for initializing their respective architecture to the EfficientNetBase init, which runs it.
class EfficientNetB0Transfer(EfficientNetTransfer):
    def __init__(self, cfg):
        super().__init__(cfg, models.efficientnet_b0)


class EfficientNetB1Transfer(EfficientNetTransfer):
    def __init__(self, cfg):
        super().__init__(cfg, models.efficientnet_b1)


class EfficientNetB2Transfer(EfficientNetTransfer):
    def __init__(self, cfg):
        super().__init__(cfg, models.efficientnet_b2)


class EfficientNetB3Transfer(EfficientNetTransfer):
    def __init__(self, cfg):
        super().__init__(cfg, models.efficientnet_b3)


class EfficientNetB4Transfer(EfficientNetTransfer):
    def __init__(self, cfg):
        super().__init__(cfg, models.efficientnet_b4)


class EfficientNetB5Transfer(EfficientNetTransfer):
    def __init__(self, cfg):
        super().__init__(cfg, models.efficientnet_b5)


class EfficientNetB6Transfer(EfficientNetTransfer):
    def __init__(self, cfg):
        super().__init__(cfg, models.efficientnet_b6)


class EfficientNetB7Transfer(EfficientNetTransfer):
    def __init__(self, cfg):
        super().__init__(cfg, models.efficientnet_b7)
