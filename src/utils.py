import torch
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')


def save_model(epochs, model, optimizer, criterion, pretrained, num_classes, bayes_type):
    """
    Function to save the trained model to disk.
    """
    model_name = model.__class__.__name__
    loss_name = criterion.__class__.__name__
    if num_classes == 2:
        tmp_dir = 'binary'
    else:
        tmp_dir = 'multi'
    try:
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, f"../results/{tmp_dir}/model{model_name}_pretrained_{pretrained}_loss_{loss_name}_bayesianType_{bayes_type}_numClass_{num_classes}.pth")
    except:
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, f"./results/{tmp_dir}/model{model_name}_pretrained_{pretrained}_loss_{loss_name}_bayesianType_{bayes_type}_numClass_{num_classes}.pth")

def save_plots(model, criterion, metrics_df, pretrained, num_classes, bayes_type):
    """
    Function to save the loss and accuracy plots to disk.
    """
    model_name = model.__class__.__name__
    loss_name = criterion.__class__.__name__

    if num_classes == 2:
        tmp_dir = 'binary'
    else:
        tmp_dir = 'multi'

    # save to csv first
    try:
        csv_save_path = f"./results/{tmp_dir}/df_model_{model_name}_pretrained_{pretrained}_loss_{loss_name}_numClass_{num_classes}_Bay_{bayes_type}.csv"
        metrics_df.to_csv(csv_save_path)
    except:
        csv_save_path = f"../results/{tmp_dir}/df_model_{model_name}_pretrained_{pretrained}_loss_{loss_name}_numClass_{num_classes}_Bay_{bayes_type}.csv"
        metrics_df.to_csv(csv_save_path)

    for m_name in metrics_df.index:
        plt.figure(figsize=(16, 9))
        y_label = m_name
        m_vals_train = metrics_df.loc[m_name, 'Train']
        m_vals_val = metrics_df.loc[m_name, 'Val']
        m_vals_test = metrics_df.loc[m_name, 'Test']
        plt.plot(m_vals_train, linestyle='dashed', color='green', label='Train')
        plt.plot(m_vals_val, linestyle='dotted', color='purple', label='Val')
        plt.plot(m_vals_test, linestyle='solid', color='gold', label='Test')
        save_path = f"./results/{tmp_dir}/plot_model_{model_name}_{pretrained}_loss_{loss_name}_numClass_{num_classes}_Bay_{bayes_type}_metric_{y_label}.jpg"
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(y_label.capitalize())
        plt.title(f"{y_label.capitalize()} for Train, Val, and Test Dataset.")
        try:
            plt.savefig(save_path)
        except:
            plt.savefig('.' + save_path)
