from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision import transforms
import torch
import pandas as pd
from model import build_effnet_model
from resnet import build_resnet_model
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

def evaluate_model_performance(model, class_names, image_label_mapping, image_size, pretrained=True, device='cpu'):
    y_true = []
    y_pred = []

    # Iterate over all the images and do forward pass.
    for image_path in image_label_mapping:
        # Get the ground truth class name from the image path.
        gt_class_name = image_label_mapping.get(image_path, "Unknown")
        gt_class_name = class_names[gt_class_name]
        image = Image.open(image_path).convert('RGB')
        # orig_image = image.copy()

        # Preprocess the image
        transform = get_test_transform(image_size=image_size, pretrained=pretrained)

        image = transform(image)
        image = torch.unsqueeze(image, 0)
        image = image.to(device)

        # Forward pass through the image.
        with torch.no_grad():
            outputs = model(image)
        outputs = outputs.softmax(1).argmax(1)
        outputs = outputs.detach().numpy()
        pred_class_name = class_names[outputs[0]]


        y_true.append(gt_class_name)
        y_pred.append(pred_class_name)
    return y_true, y_pred


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
    num_classes = 2
    IMAGE_SIZE = 224
    bayes_type = "all"

    # read data
    # Load the training and validation datasets.
    _, _, test_data, _, _, _ = get_data_loaders(data_dir=data_dir, num_classes = num_classes)

    # Class names.
    if num_classes == 2:
        class_names = ['NORMAL', 'PNEUMONIA']
    else:
        class_names = ['NORMAL', 'VIRUS PNEUMONIA', 'BACTERIA PNEUMONIA']

    # Load the trained model.
    # model = build_effnet_model(pretrained=True, fine_tune=False, num_classes=num_classes, bayes_type=bayes_type)
    # if num_classes == 2:
    #     try:
    #         checkpoint = torch.load('../results/binary/modelEfficientNet_pretrained_True_loss_CrossEntropyLoss_bayesianType_all_numClass_2.pth', map_location=DEVICE)
    #     except:
    #         checkpoint = torch.load('./results/binary/modelEfficientNet_pretrained_True_loss_CrossEntropyLoss_bayesianType_all_numClass_2.pth', map_location=DEVICE)
    # else:
    #     try:
    #         checkpoint = torch.load('../results/multi/modelEfficientNet_pretrained_True_loss_CrossEntropyLoss_bayesianType__numClass_3.pth', map_location=DEVICE)
    #     except:
    #         checkpoint = torch.load('./results/multi/modelEfficientNet_pretrained_True_loss_CrossEntropyLoss_bayesianType__numClass_3.pth', map_location=DEVICE)
    model = build_resnet_model(pretrained=True, fine_tune=False, num_classes=num_classes, bayes_type=bayes_type)
    if num_classes == 2:
        try:
            checkpoint = torch.load('../results/binary/modelResNet_pretrained_True_loss_CrossEntropyLoss_bayesianType_all_numClass_2.pth', map_location=DEVICE)
        except:
            checkpoint = torch.load('./results/binary/modelResNet_pretrained_True_loss_CrossEntropyLoss_bayesianType_all_numClass_2.pth', map_location=DEVICE)
    else:
        try:
            checkpoint = torch.load('../results/multi/modelResNet_pretrained_True_loss_CrossEntropyLoss_bayesianType_all_numClass_3.pth', map_location=DEVICE)
        except:
            checkpoint = torch.load('./results/multi/modelResNet_pretrained_True_loss_CrossEntropyLoss_bayesianType_all_numClass_3.pth', map_location=DEVICE)

    print('Loading trained model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load ground truth labels from file.
    # image_label_mapping = pd.DataFrame(test_data).iloc[:, 0]
    image_label_mapping = {item[0]: item[1] for item in test_data}
    labels_df = pd.DataFrame(test_data).iloc[:, 1]
    labels_df = labels_df.sample(5)

    # Usage
    y_true, y_pred = evaluate_model_performance(model=model,
                                                class_names=class_names,
                                                image_label_mapping=image_label_mapping,
                                                image_size=IMAGE_SIZE,
                                                device=DEVICE)
    plot_classification_results(y_true=y_true, y_pred=y_pred, class_names=class_names)
