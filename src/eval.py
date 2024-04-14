from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision import transforms
import torch
import pandas as pd
from model import build_effnet_model


def evaluate_model_performance(model, class_names, image_label_mapping, image_size, device='cpu'):
    y_true = []
    y_pred = []

    # Iterate over all the images and do forward pass.
    for image_path in image_label_mapping.keys():
        # Get the ground truth class name from the image path.
        gt_class_name = image_label_mapping.get(image_path, "Unknown")

        image = cv2.imread(image_path)
        # orig_image = image.copy()

        # Preprocess the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        image = image.to(device)

        # Forward pass through the image.
        # TODO: Add forward pass time (computational cost)
        # see: https://debuggercafe.com/pytorch-pretrained-efficientnet-model-image-classification/
        with torch.no_grad():
            outputs = model(image)
        outputs = outputs.softmax(1).argmax(1)
        outputs = outputs.detach().numpy()
        pred_class_name = class_names[np.argmax(outputs[0])]
        # print(f"GT: {gt_class_name}, Pred: {pred_class_name}")
        # Annotate the image with ground truth.
        # cv2.putText(
        #     orig_image, f"GT: {gt_class_name}",
        #     (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
        #     1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA
        # )
        # # Annotate the image with prediction.
        # cv2.putText(
        #     orig_image, f"Pred: {pred_class_name}",
        #     (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
        #     1.0, (100, 100, 225), 2, lineType=cv2.LINE_AA
        # )
        # cv2.imshow('Result', orig_image)
        # # cv2.waitKey(0)
        # cv2.imwrite(f"../results/tests/{image_path}__{str(gt_class_name)}.png", orig_image)

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
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # Compute ROC curve and AUC
    # Assuming binary classification
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


if __name__ == '__main__':
    # Constants.
    DATA_PATH = '../data/imgs'
    CSV_PATH = '../data/panel_data.csv'
    IMAGE_SIZE = 224
    DEVICE = 'cpu'
    target_col = 'Target'  # annotationNumber

    # Class names.
    class_names = [1, 2, 3, 4, 5] if target_col == "annotationNumber" else [0, 1]

    # Load the trained model.
    model = build_effnet_model(pretrained=True, fine_tune=False, num_classes=len(class_names))
    checkpoint = torch.load('../results/modelEfficientNet_pretrained_True_loss_CrossEntropyLoss.pth', map_location=DEVICE)
    print('Loading trained model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load ground truth labels from CSV file.
    labels_df = pd.read_csv(CSV_PATH)
    labels_df = labels_df.sample(20)
    labels_df['ImagePath'] = labels_df['ImagePath'].apply(lambda x: "../" + x)
    image_label_mapping = dict(zip(labels_df['ImagePath'], labels_df[target_col]))

    # Usage
    y_true, y_pred = evaluate_model_performance(model=model,
                                                class_names=class_names,
                                                image_label_mapping=image_label_mapping,
                                                image_size=IMAGE_SIZE,
                                                device=DEVICE)

    plot_classification_results(y_true=y_true, y_pred=y_pred, class_names=class_names)
