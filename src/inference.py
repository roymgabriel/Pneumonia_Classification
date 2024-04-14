import torch
import cv2
import numpy as np
import glob as glob
import os
from model import build_effnet_model
from torchvision import transforms
import pandas as pd

# Constants.
DATA_PATH = '../data/imgs'
CSV_PATH = '../data/panel_data.csv'
IMAGE_SIZE = 224
DEVICE = 'cpu'
target_col = 'Target'  # annotationNumber

# Class names.
class_names = [1, 2, 3, 4, 5] if target_col == "annotationNumber" else [0, 1]
pretrained = True
num_classes = len(class_names)
bayes_last = False
loss_name = "CrossEntropyLoss"

# Load the trained model.
model = build_effnet_model(pretrained=pretrained, fine_tune=False, num_classes=num_classes, bayes_last=bayes_last)
model_name = model.__class__.__name__

# checkpoint = torch.load('../results/modelEfficientNet_pretrained_True_loss_CrossEntropyLoss_numClass_2.pth', map_location=DEVICE)
checkpoint = torch.load(f'../results/model{model_name}_pretrained_{pretrained}_loss_{loss_name}_bayesianLast_{bayes_last}_numClass_{num_classes}.pth',
                        map_location=DEVICE)

print('Loading trained model weights...')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load ground truth labels from CSV file.
labels_df = pd.read_csv(CSV_PATH)
labels_df = labels_df.sample(100)
labels_df['ImagePath'] = labels_df['ImagePath'].apply(lambda x: "../" + x)
image_label_mapping = dict(zip(labels_df['ImagePath'], labels_df[target_col]))

# Get all the eval.py image paths.
# all_image_paths = glob.glob(f"{DATA_PATH}/*")
# Iterate over all the images and do forward pass.
for image_path in image_label_mapping.keys():
    # Get the ground truth class name from the image path.
    # image_name = os.path.basename(image_path)
    gt_class_name = image_label_mapping.get(image_path, "Unknown")  # Get label from mapping or set to "Unknown" if
    # not found

    image = cv2.imread(image_path)
    orig_image = image.copy()

    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
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
    cv2.imwrite(f"../results/tests/{image_path}__{str(gt_class_name)}.png", orig_image)