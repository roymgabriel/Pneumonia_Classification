import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm
from model import build_effnet_model
from datasets import get_data_loaders
from utils import save_model, save_plots
from torcheval import metrics
from torcheval.metrics.toolkit import clone_metrics
import pandas as pd

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--epochs', type=int, default=2,
    help='Number of epochs to train our network for'
)

parser.add_argument(
    '-b', '--batchsize', type=int, default=16,
    help='Batch size for data loaders'
)

parser.add_argument(
    '-nw', '--numworkers', type=int, default=3,
    help='Number of workers for data loaders'
)


parser.add_argument(
    '-pt', '--pretrained', action='store_true', default=True,
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
    '-nc', '--numclasses', type=int, default=2,
    help='Whether binary or multi-class classifcation'
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

def log_eval(epoch_idx, train_stat, val_stat, test_stat, task_index=0):


    print(f"Epoch:{epoch_idx} =========================== \n")
    meter_list = [train_stat[task_index], val_stat[task_index], test_stat[task_index]]

    # Log to console
    output_message = ""
    for dataset_index, (meter, dataset_type) in enumerate(zip(meter_list, ["Train", "Val", "Test"])):
        for metric_index, eval in enumerate(metrics_str):
            output_message += f"{dataset_type} {eval}:{meter_list[dataset_index][metric_index]:.4f} | "
            metric_vals[dataset_type][eval].append(meter_list[dataset_index][metric_index])
        output_message += "\n"
    print(output_message)
    print("\n")

    return metric_vals


def calc_loss(logit, ys):
    losses = []

    y = ys.reshape(-1)
    loss = 0
    l = criterion(logit, y)
    loss += l
    losses.append(l)

    return losses, loss

def evaluate(meters, logits, ys, losses, task_index=0):
    logit = logits.detach()
    y = ys.reshape(-1).detach()
    loss = losses[task_index].detach()

    # Store the values needed to calculate the mean of losses [0], accuracy [1], precision [2], recall [3], and F1 score [4] later.
    if num_classes == 2:
        meters[task_index][0].update(loss)
        meters[task_index][1].update(logit.softmax(1).argmax(1), y)
        meters[task_index][2].update(logit.softmax(1).argmax(1), y)
        meters[task_index][3].update(logit.softmax(1).argmax(1), y)
        meters[task_index][4].update(logit.softmax(1).argmax(1), y)
        meters[task_index][5].update(logit.softmax(1).argmax(1), y)
        meters[task_index][6].update(logit.softmax(1).argmax(1), y)
    else:
        meters[task_index][0].update(loss)
        meters[task_index][1].update(logit, y)
        meters[task_index][2].update(logit, y)
        meters[task_index][3].update(logit, y)
        meters[task_index][4].update(logit, y)
        meters[task_index][5].update(logit, y)
        meters[task_index][6].update(logit, y)

    return meters


# Training function.
def train(model, trainloader, optimizer, criterion):
    model.train()

    # Ensure that the template eval_metrics are empty before cloning them. Nothing should be changing them,
    # but just to be certain.
    for i in range(num_eval_metrics):
        eval_metrics[i].reset()

    # This creates multiple copies of metrics for the multiple tasks.
    # Each clone of eval_metrics creates a list of new torcheval metric instances, which each
    # store all the values needed to calculate their metric for one task in this epoch.
    # The metric is only calculated once .compute() is called on the metric instance.
    train_meters = [clone_metrics(eval_metrics) for i in range(1)]


    print('Training')
    # train_running_loss = 0.0
    # train_running_correct = 0
    counter = 0
    def foo_wrapper(model, train_meters, counter):
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

            # # Calculate the loss.
            # loss = criterion(outputs, labels)
            # train_running_loss += loss.item()
            # # Calculate the accuracy.
            # _, preds = torch.max(outputs.data, 1)
            # train_running_correct += (preds == labels).sum().item()

            losses, loss = calc_loss(outputs, labels)
            (loss).backward()
            optimizer.step()
            train_meters = evaluate(train_meters, outputs, labels, losses)

        return train_meters

    train_meters = foo_wrapper(model=model, train_meters=train_meters, counter=counter)

    # Loss and accuracy for the complete epoch.
    # epoch_loss = train_running_loss / counter
    # epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return train_meters


# Validation function.
def validate(model, testloader, criterion):
    model.eval()
    val_meters = [clone_metrics(eval_metrics) for i in range(1)]
    print('Testing')
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
            # loss = criterion(outputs, labels)
            # valid_running_loss += loss.item()
            # # Calculate the accuracy.
            # _, preds = torch.max(outputs.data, 1)
            # valid_running_correct += (preds == labels).sum().item()

            losses, loss = calc_loss(outputs, labels)
            val_meters = evaluate(val_meters, outputs, labels, losses)


    # Loss and accuracy for the complete epoch.
    # epoch_loss = valid_running_loss / counter
    # epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return val_meters


if __name__ == '__main__':
    # torch.set_default_device("mps")
    # Load the training and validation datasets.
    # dataset_train, dataset_valid, dataset_test,\
    #     dataset_classes = get_datasets(is_binary=True, pretrained=args['pretrained'])
    # print(f"[INFO]: Number of training images: {len(dataset_train)}")
    # print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    # print(f"[INFO]: Number of testing images: {len(dataset_test)}")
    # print(f"[INFO]: Class names: {dataset_classes}\n")

    # Learning_parameters.
    lr = args['learning_rate']
    epochs = args['epochs']
    device = args['device']
    bayesian = args['bayesian']
    num_classes = args['numclasses']
    fine_tune=args['finetune']
    pretrained=args['pretrained']
    batch_size=args['batchsize']
    num_workers=args['numworkers']

    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}")
    print(f"Bayesian Last Layer Only: {bayesian}")
    print(f"Number of Classes: {num_classes}")
    print(f"Fine Tuned: {fine_tune}")
    print(f"Pretrained: {pretrained}")
    print(f"Batch Size: {batch_size}")
    print(f"Number of Workers: {num_workers}\n")

    data_dir = "./data/chest_xray/"

    # Load the training and validation data loaders.
    train_loader, valid_loader, test_loader = get_data_loaders(data_dir=data_dir, batch_size=batch_size,\
        num_workers=num_workers, num_classes=num_classes, image_size=224, pretrained=pretrained)

    model = build_effnet_model(
        pretrained=pretrained,
        fine_tune=fine_tune,
        bayes_last=bayesian,
        num_classes=num_classes
    ).to(device)

    OPTIM = torch.optim.SGD
    OPTIM_PARAMETERS = {
        'lr': lr,
        'weight_decay': 0.1,
        'momentum': 0.9
    }
    LR_SCHEDULER = torch.optim.lr_scheduler.CosineAnnealingLR
    LR_SCHEDULER_PARAMETERS = {
        'T_max': epochs, 'eta_min': 0, 'last_epoch': - 1
    }


    # Optimizer.
    optimizer = OPTIM(model.parameters(), **OPTIM_PARAMETERS)
    lr_scheduler = LR_SCHEDULER(optimizer, **LR_SCHEDULER_PARAMETERS)

    # Loss function.
    criterion = nn.CrossEntropyLoss()


    if num_classes == 2:
        eval_metrics = [
            metrics.Mean(device=device),
            metrics.BinaryAccuracy(device=device),
            metrics.BinaryPrecision(device=device),
            metrics.BinaryRecall(device=device),
            metrics.BinaryF1Score(device=device),
            metrics.BinaryAUROC(device=device),
            metrics.BinaryAUPRC(device=device)
        ]
    else:
        eval_metrics = [
            metrics.Mean(device=device),
            metrics.MulticlassAccuracy(device=device, average="macro", num_classes=num_classes),
            metrics.MulticlassPrecision(device=device, average="macro", num_classes=num_classes),
            metrics.MulticlassRecall(device=device, average="macro", num_classes=num_classes),
            metrics.MulticlassF1Score(device=device, average="macro", num_classes=num_classes),
            metrics.MulticlassAUROC(device=device, average="macro", num_classes=num_classes),
            metrics.MulticlassAUPRC(device=device, average="macro", num_classes=num_classes)
        ]
    num_eval_metrics = len(eval_metrics)


    metrics_str = ["loss", "accuracy", "precision", "recall", "F1", "AUC ROC", "AUC PRC"]
    metric_vals = {'Train': {}, 'Val': {}, 'Test': {}}
    for key in metric_vals:
        metric_vals[key] = {metric: [] for metric in metrics_str}


    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Lists to keep track of losses and accuracies.
    # train_loss, valid_loss = [], []
    # train_acc, valid_acc = [], []
    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
        train_meters = train(model, train_loader, optimizer, criterion)
        val_meters = validate(model, valid_loader, criterion)
        test_meters = validate(model, test_loader, criterion)

        # train_loss.append(train_epoch_loss)
        # valid_loss.append(valid_epoch_loss)
        # train_acc.append(train_epoch_acc)
        # valid_acc.append(valid_epoch_acc)
        # print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        # print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        # print('-' * 50)
        # time.sleep(5)

        # Compute all the metrics by calling .compute() on each of the metric instances,
        # and store them in a list with the same structure as train_meters, val_meters, test_meters.
        task_idx = 0
        train_compute = [[train_meters[0][metric_idx].compute().cpu().item() for metric_idx in
                              range(num_eval_metrics)] for task_idx in range(1)]
        val_compute = [[val_meters[0][metric_idx].compute().cpu().item() for metric_idx in
                        range(num_eval_metrics)] for task_idx in range(1)]

        test_compute = [[test_meters[0][metric_idx].compute().cpu().item() for metric_idx in
                        range(num_eval_metrics)] for task_idx in range(1)]

        metric_vals = log_eval(epoch_idx=epoch, train_stat=train_compute, val_stat=val_compute, test_stat=test_compute)

        # Get the current learning rate
        current_lr = lr_scheduler.get_last_lr()
        print(f"\nCurrent Learning Rate: {current_lr}\n")

        if lr_scheduler:
            lr_scheduler.step()

    # change metric vals to pandas dictionary
    metrics_df = pd.DataFrame(metric_vals)

    # Save the trained model weights.
    save_model(epochs=epochs,
               model=model,
               optimizer=optimizer,
               criterion=criterion,
               pretrained=pretrained,
               num_classes=num_classes,
               bayes_last=bayesian)

    # Save the loss and accuracy plots.
    save_plots(model=model,
               criterion=criterion,
               metrics_df=metrics_df,
               pretrained=pretrained,
               num_classes=num_classes,
               bayes_last=bayesian)
    print('TRAINING COMPLETE\n')
