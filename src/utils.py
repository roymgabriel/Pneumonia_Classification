import torch
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')


def save_model(epochs, model, optimizer, criterion, pretrained, num_classes, bayes_last):
    """
    Function to save the trained model to disk.
    """
    model_name = model.__class__.__name__
    loss_name = criterion.__class__.__name__
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, f"../results/model{model_name}_pretrained_{pretrained}_loss_{loss_name}_bayesianLast_{bayes_last}_numClass_{num_classes}.pth")

def save_plots(model, criterion, train_acc, valid_acc, train_loss, valid_loss, pretrained, num_classes, bayes_last):
    """
    Function to save the loss and accuracy plots to disk.
    """
    model_name = model.__class__.__name__
    loss_name = criterion.__class__.__name__

    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validation accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"../results/model_{model_name}_loss_{loss_name}_accuracy_pretrained_{pretrained}_bayesLast_{bayes_last}_numClass_{num_classes}.png")

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validation loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"../results/model_{model_name}_loss_{loss_name}_pretrained_{pretrained}_bayesLast_{bayes_last}_classnum_{num_classes}.png")
