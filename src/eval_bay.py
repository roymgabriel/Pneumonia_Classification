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
    print('EVALUATING MODEL PERFORMANCE')
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

    # Adjusted code to create a separate plot for each image

    for idx, (image_path, means, stds) in enumerate(results):
        # Set up the plot
        fig, ax = plt.subplots(figsize=(10, 6))  # Use subplots instead of figure for proper axes handling
        x = np.arange(len(class_names))  # Adjust x to not offset for each image
        ax.bar(x, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10)
        ax.set_ylabel('Probability')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.set_title(f'Image {idx + 1} Class Probabilities with Error Bars')
        ax.yaxis.grid(True)

        # Rotate the x-axis labels so they don't overlap
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Save or show the plot
        plt.tight_layout()
        plt.savefig(f'../image_{idx + 1}_probabilities_with_error_bars.png')
        plt.show()




if __name__ == '__main__':
    # Constants.
    data_dir = "./data/chest_xray/"
    DEVICE = 'mps'
    num_classes = 3
    IMAGE_SIZE = 224
    bayes_type = "last"

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
            checkpoint = torch.load(f'../results/binary/modelEfficientNet_pretrained_True_loss_CrossEntropyLoss_bayesianType_{bayes_type}_numClass_2.pth', map_location=DEVICE)
        except:
            checkpoint = torch.load(f'./results/binary/modelEfficientNet_pretrained_True_loss_CrossEntropyLoss_bayesianType_{bayes_type}_numClass_2.pth', map_location=DEVICE)
    else:
        try:
            checkpoint = torch.load(f'../results/multi/modelEfficientNet_pretrained_True_loss_CrossEntropyLoss_bayesianType_{bayes_type}_numClass_3.pth', map_location=DEVICE)
        except:
            checkpoint = torch.load(f'./results/multi/modelEfficientNet_pretrained_True_loss_CrossEntropyLoss_bayesianType_{bayes_type}_numClass_3.pth', map_location=DEVICE)

    print('Loading trained model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    print('DONE')
    model.to(DEVICE)
    model.eval()

    # Assuming test_data is a list of tuples, where each tuple is (image_path, label)
    test_data_df = pd.DataFrame(test_data, columns=['image_path', 'label'])

    # Sample 5 entries from the DataFrame
    sampled_data = test_data_df.sample(5)

    # Create the image_label_mapping dictionary from the sampled data
    image_label_mapping = dict(zip(sampled_data['image_path'], sampled_data['label']))

    K = 100  # Number of forward passes
    results = evaluate_model_performance(model=model, class_names=class_names, image_label_mapping=image_label_mapping, image_size=IMAGE_SIZE, K=K, device=DEVICE)
    plot_probabilities_with_error_bars(results, class_names)
