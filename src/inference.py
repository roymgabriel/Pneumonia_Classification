import torch
import cv2
import numpy as np
import glob as glob
import os
from model import build_effnet_model
from torchvision import transforms
import pandas as pd
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

if __name__ == '__main__':


    # Constants.

    data_dir = "./data/chest_xray/"
    DEVICE = 'cpu'
    num_classes = 2
    image_size = 224

    # Class names.
    # Load the training and validation datasets.
    _, _, test_data, _, _, _ = get_data_loaders(data_dir=data_dir)

    # Class names.
    if num_classes == 2:
        class_names = ['NORMAL', 'PNEUMONIA']
    else:
        class_names = ['NORMAL', 'VIRUS PNEUMONIA', 'BACTERIA PNEUMONIA']

    pretrained = True
    num_classes = len(class_names)
    bayes_last = False
    loss_name = "CrossEntropyLoss"
# Class names.
class_names = [1, 2, 3, 4, 5] if target_col == "annotationNumber" else [0, 1]
pretrained = True
num_classes = len(class_names)
bayes_type = 'last'
loss_name = "CrossEntropyLoss"

    # Load the trained model.
    model = build_effnet_model(pretrained=True, fine_tune=False, num_classes=num_classes, bayes_last=False)
    if num_classes == 2:
        try:
            checkpoint = torch.load('../results/binary/modelEfficientNet_pretrained_True_loss_CrossEntropyLoss_bayesianLast_False_numClass_2.pth', map_location=DEVICE)
        except:
            checkpoint = torch.load('./results/binary/modelEfficientNet_pretrained_True_loss_CrossEntropyLoss_bayesianLast_False_numClass_2.pth', map_location=DEVICE)
    else:
        try:
            checkpoint = torch.load('../results/multi/modelEfficientNet_pretrained_True_loss_CrossEntropyLoss_bayesianLast_False_numClass_3.pth', map_location=DEVICE)
        except:
            checkpoint = torch.load('./results/multi/modelEfficientNet_pretrained_True_loss_CrossEntropyLoss_bayesianLast_False_numClass_3.pth', map_location=DEVICE)
# Load the trained model.
model = build_effnet_model(pretrained=pretrained, fine_tune=False, num_classes=num_classes, bayes_type=bayes_type)
model_name = model.__class__.__name__

# checkpoint = torch.load('../results/modelEfficientNet_pretrained_True_loss_CrossEntropyLoss_numClass_2.pth', map_location=DEVICE)
checkpoint = torch.load(f'../results/model{model_name}_pretrained_{pretrained}_loss_{loss_name}_bayesianType_{bayes_type}_numClass_{num_classes}.pth',
                        map_location=DEVICE)

    print('Loading trained model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    model_name = model.__class__.__name__
    model.eval()

    # Load ground truth labels from file.
    image_label_mapping = pd.DataFrame(test_data)
    image_label_mapping = image_label_mapping.sample(5)

    # Get all the eval.py image paths.
    # all_image_paths = glob.glob(f"{DATA_PATH}/*")
    # Iterate over all the images and do forward pass.
    for i, d in enumerate(image_label_mapping.values):
        # Get the ground truth class name from the image path.
        image_path = d[0]
        gt_class_name = class_names[d[1]]

        image = Image.open(image_path).convert('RGB')
        orig_image = image.copy()
        orig_image = np.asarray(orig_image).astype(np.float32)

        # Preprocess the image
        transform = get_test_transform(image_size=image_size, pretrained=pretrained)

        image = transform(image)
        image = torch.unsqueeze(image, 0)
        image = image.to(DEVICE)

        # Forward pass through the image.
        with torch.no_grad():
            outputs = model(image)
        outputs = outputs.softmax(1).argmax(1)
        outputs = outputs.detach().numpy()
        pred_class_name = class_names[np.argmax(outputs[0])]

        print(f"GT: {gt_class_name}, Pred: {pred_class_name}")
        # Annotate the image with ground truth.
        cv2.putText(
            orig_image, f"GT: {gt_class_name}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA
        )
        # Annotate the image with prediction.
        cv2.putText(
            orig_image, f"Pred: {pred_class_name}",
            (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (100, 100, 225), 2, lineType=cv2.LINE_AA
        )
        cv2.imshow('Result', orig_image)
        # cv2.waitKey(0)
        if num_classes == 2:
            cv2.imwrite(f"../results/binary/tests/{image_path}__{str(gt_class_name)}_{i}.png", orig_image)
            cv2.imwrite(f"./results/binary/tests/{image_path}__{str(gt_class_name)}_{i}.png", orig_image)

        else:
            cv2.imwrite(f"../results/multi/tests/{image_path}__{str(gt_class_name)}_{i}.png", orig_image)
            cv2.imwrite(f"./results/multi/tests/{image_path}__{str(gt_class_name)}_{i}.png", orig_image)
