import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from sklearn.model_selection import train_test_split
from PIL import Image

# Required constants.
ROOT_DIR = '../data/panel_data_rsna.csv'
VALID_SPLIT = None
TEST_SPLIT = 0.1
IMAGE_SIZE = 224  # Image size of resize when applying transforms.
BATCH_SIZE = 16
NUM_WORKERS = 3  # Number of parallel processes for data preparation.


# Training transforms
def get_train_transform(IMAGE_SIZE, pretrained):
    train_transform = transforms.Compose([
        transforms.Resize((image_size_0, image_size_0)),
        transforms.Grayscale(3),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.RandomOrder([
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))], 1),
            transforms.RandomApply([transforms.RandomAffine(degrees=(-10, 10))], 1),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, scale=(.98, 1.02))], 1),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=(-5, 5))], 1),
        ]),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return train_transform


# Validation transforms
def get_test_transform(IMAGE_SIZE, pretrained):
    test_transform = transforms.Compose([
        transforms.Resize((image_size_0, image_size_0)),
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


# Assuming you have a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe, target_col, transform=None):
        self.dataframe = dataframe
        self.transform = transform

        # fix image path in dataframe
        self.dataframe['patientId'] = self.dataframe['patientId'].apply(lambda x: "../data/imgs/train/" + x + ".jpg")

        # define target col name
        self.target_col = target_col

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Assuming 'ImagePath' column contains paths to images
        img_path = self.dataframe.iloc[idx]['patientId']
        image = Image.open(img_path).convert("RGB")

        # Assuming you have other columns like labels
        label = self.dataframe.iloc[idx][self.target_col]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_datasets(pretrained, is_binary, random_state=42, sample_num=None):
    """
    Function that obtains the datasets
    :type random_state: int
    """
    target_col = 'Target' if is_binary else 'class'
    panel_data = pd.read_csv("../data/panel_data_rsna.csv")
    if sample_num is not None:
        panel_data = panel_data.sample(sample_num) # make it shorter so it runs faster
    train_data, test_data = train_test_split(panel_data, test_size=TEST_SPLIT, random_state=random_state)

    # Further split training data into training and validation sets (90% training, 10% validation)
    train_data, val_data = train_test_split(train_data, test_size=VALID_SPLIT, random_state=random_state)

    # Define transforms
    train_transform = get_train_transform(IMAGE_SIZE, pretrained=pretrained)  # Assuming training from scratch
    test_transform = get_test_transform(IMAGE_SIZE, pretrained=pretrained)  # Assuming training from scratch

    # Create datasets and data loaders
    train_dataset = CustomDataset(train_data, target_col=target_col, transform=train_transform)
    valid_dataset = CustomDataset(val_data, target_col=target_col, transform=test_transform)
    test_dataset = CustomDataset(test_data, target_col=target_col, transform=test_transform)
    return train_dataset, valid_dataset, test_dataset, panel_data[target_col].unique()


def get_data_loaders(train_dataset, valid_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, valid_loader, test_loader
