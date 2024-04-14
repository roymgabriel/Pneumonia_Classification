import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm
from model import build_effnet_model
from datasets import get_datasets, get_data_loaders
from utils import save_model, save_plots

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--epochs', type=int, default=20,
    help='Number of epochs to train our network for'
)
parser.add_argument(
    '-pt', '--pretrained', action='store_true',
    help='Whether to use pretrained weights or not'
)

parser.add_argument(
    '-ft', '--finetune', action='store_true', default=False,
    help='Whether to finetune model or not'
)

parser.add_argument(
    '-by', '--bayesian', action='store_true', default=False,
    help='Whether to make only last layer bayesian or every convolution layer including last linear layer'
)

parser.add_argument(
    '-d', '--device', type=str, default='mps',
    help='what device to use'
)

parser.add_argument(
    '-lr', '--learning-rate', type=float,
    dest='learning_rate', default=0.0001,
    help='Learning rate for training the model'
)
args = vars(parser.parse_args())


# Training function.
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    def foo_wrapper(model, train_running_loss, train_running_correct, counter):
        """
        Wrapper function for mps devices not to crash with error:
        -[IOGPUMetalCommandBuffer validate]:216: failed assertion `commit command buffer with uncommitted encoder'
        :param model:
        :return:
        """
        for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        # for i, data in enumerate(trainloader):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # Forward pass.
            outputs = model(image)

            # Calculate the loss.
            loss = criterion(outputs, labels)
            train_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            train_running_correct += (preds == labels).sum().item()
            # Backpropagation
            loss.backward()
            # Update the weights.
            optimizer.step()
        return train_running_loss, train_running_correct, counter

    train_running_loss, train_running_correct, counter = foo_wrapper(model, train_running_loss, train_running_correct, counter)

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


# Validation function.
def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc


if __name__ == '__main__':
    # torch.set_default_device("mps")
    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_test,\
        dataset_classes = get_datasets(is_binary=True, pretrained=args['pretrained'])
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO]: Number of testing images: {len(dataset_test)}")
    print(f"[INFO]: Class names: {dataset_classes}\n")
    # Load the training and validation data loaders.
    # TODO: Add validation in training
    train_loader, valid_loader, _ = get_data_loaders(dataset_train, dataset_valid, dataset_test)
    # Learning_parameters.
    lr = args['learning_rate']
    epochs = args['epochs']
    device = args['device']
    bayesian = args['bayesian']
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")
    print(f"Bayesian Last Layer Only: {bayesian}\n")
    model = build_effnet_model(
        pretrained=args['pretrained'],
        fine_tune=args['finetune'],
        bayes_last=args['bayesian'],
        num_classes=len(dataset_classes)
    ).to(device)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Loss function.
    criterion = nn.CrossEntropyLoss()
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader,
                                                  optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,
                                                     criterion)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-' * 50)
        time.sleep(5)

    # Save the trained model weights.
    save_model(epochs=epochs,
               model=model,
               optimizer=optimizer,
               criterion=criterion,
               pretrained=args['pretrained'],
               num_classes=len(dataset_classes),
               bayes_last=args['bayesian'])
    # Save the loss and accuracy plots.
    save_plots(model=model,
               criterion=criterion,
               train_acc=train_acc,
               valid_acc=valid_acc,
               train_loss=train_loss,
               valid_loss=valid_loss,
               pretrained=args['pretrained'],
               num_classes=len(dataset_classes),
               bayes_last=args['bayesian'])
    print('TRAINING COMPLETE\n')
