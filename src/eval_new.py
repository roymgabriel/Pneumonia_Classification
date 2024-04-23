from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision import transforms
import torch
import pandas as pd
from model import build_effnet_model
from datasets import get_data_loaders
from PIL import Image


# Validation transforms
def get_test_transform(image_size, pretrained):
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return test_transform


# Image normalization transforms.
def normalize_transform(pretrained):
    if pretrained:  # Normalization for pre-trained weights.
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    else:  # Normalization when training from scratch.
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    return normalize

def evaluate_model_performance(model, class_names, image_label_mapping, image_size, K, pretrained=True, device='cpu'):
    results = []

    transform = get_test_transform(image_size=image_size, pretrained=pretrained)

    # Iterate over all the images and perform K forward passes
    for image_path in image_label_mapping:
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        image = image.to(device)

        # Collect softmax probabilities for K forward passes
        softmax_outputs = []
        with torch.no_grad():
            for _ in range(K):
                output = model(image).softmax(1)
                softmax_outputs.append(output.cpu().numpy())

        # Calculate mean and standard deviation across the K passes
        softmax_outputs = np.array(softmax_outputs)
        mean_probabilities = softmax_outputs.mean(axis=0).flatten()
        std_dev_probabilities = softmax_outputs.std(axis=0).flatten()

        results.append((image_path, mean_probabilities, std_dev_probabilities))

    return results

def plot_probabilities_with_error_bars(results, class_names):
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Loop through results to plot each set of probabilities with error bars
    for idx, (image_path, means, stds) in enumerate(results):
        x = np.arange(len(class_names)) + idx * (len(class_names) + 2)  # Offset x position for each image
        ax.bar(x, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10, label=f'Image {idx + 1}')

    ax.set_ylabel('Probability')
    ax.set_xticks(np.arange(len(class_names)) + len(results) / 2 * (len(class_names) + 2) - len(class_names))
    ax.set_xticklabels(class_names)
    ax.set_title('Class Probabilities with Standard Deviation Error Bars')
    ax.yaxis.grid(True)
    ax.legend()

    # Save or show the plot
    plt.tight_layout()
    plt.show()


def plot_classification_results(y_true, y_pred, class_names):
    """
    Plots classification report and other metrics
    :param y_true:
    :param y_pred:
    :return:
    """
    # Compute classification metrics
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=max(class_names))
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()


    # Compute Precision-Recall curve and AUC
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=max(class_names))
    prc_auc = auc(recall, precision)

    plt.figure()
    lw = 2
    plt.plot(recall, precision, color='darkorange',
            lw=lw, label='PR curve (area = %0.2f)' % prc_auc)
    plt.fill_between(recall, precision, alpha=0.2, color='darkorange', lw=lw)  # Optional: fill under curve
    plt.plot([0, 1], [max(y_true.mean(), 1e-6)], linestyle='--', color='navy', lw=lw)  # Horizontal line at class balance
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()


if __name__ == '__main__':
    # Constants.
    data_dir = "./data/chest_xray/"
    DEVICE = 'cpu'
    num_classes = 3
    IMAGE_SIZE = 224
    bayes_type = "all"

    # read data
    # Load the training and validation datasets.
    _, _, test_data, _, _, _ = get_data_loaders(data_dir=data_dir)

    # Class names.
    if num_classes == 2:
        class_names = ['NORMAL', 'PNEUMONIA']
    else:
        class_names = ['NORMAL', 'VIRUS PNEUMONIA', 'BACTERIA PNEUMONIA']

    # Load the trained model.
    model = build_effnet_model(pretrained=True, fine_tune=False, num_classes=num_classes, bayes_type=bayes_type)
    if num_classes == 2:
        try:
            checkpoint = torch.load('../results/binary/modelEfficientNet_pretrained_True_loss_CrossEntropyLoss_bayesianType__numClass_2.pth', map_location=DEVICE)
        except:
            checkpoint = torch.load('./results/binary/modelEfficientNet_pretrained_True_loss_CrossEntropyLoss_bayesianType__numClass_2.pth', map_location=DEVICE)
    else:
        try:
            checkpoint = torch.load('../results/multi/modelEfficientNet_pretrained_True_loss_CrossEntropyLoss_bayesianType_all_numClass_3.pth', map_location=DEVICE)
        except:
            checkpoint = torch.load('./results/multi/modelEfficientNet_pretrained_True_loss_CrossEntropyLoss_bayesianType_all_numClass_3.pth', map_location=DEVICE)

    print('Loading trained model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load ground truth labels from file.
    # image_label_mapping = pd.DataFrame(test_data).iloc[:, 0]
    image_label_mapping = {item[0]: item[1] for item in test_data}
    labels_df = pd.DataFrame(test_data).iloc[:, 1]
    labels_df = labels_df.sample(5)

    # Usage
    # y_true, y_pred = evaluate_model_performance(model=model,
    #                                             class_names=class_names,
    #                                             image_label_mapping=image_label_mapping,
    #                                             image_size=IMAGE_SIZE,
    #                                             device=DEVICE)

    # plot_classification_results(y_true=y_true, y_pred=y_pred, class_names=class_names)
    K = 10  # Number of forward passes
    results = evaluate_model_performance(model=model, class_names=class_names, image_label_mapping=image_label_mapping, image_size=IMAGE_SIZE, K=K, device=DEVICE)
    plot_probabilities_with_error_bars(results, class_names)
