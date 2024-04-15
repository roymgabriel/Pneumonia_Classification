import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from sklearn.model_selection import train_test_split
from PIL import Image

# # Required constants.
# ROOT_DIR = '../data/chest_xray.csv'
# VALID_SPLIT = None
# TEST_SPLIT = 0.1
# IMAGE_SIZE = 224  # Image size of resize when applying transforms.
# BATCH_SIZE = 16
# NUM_WORKERS = 3  # Number of parallel processes for data preparation.


# Training transforms
def get_train_transform(image_size, pretrained):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
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


class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_classes=2):
        '''
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            num_classes (int): Number of classes to differentiate (2 for binary, 3 for multi-class).
        '''
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = num_classes
        self.data = []
        self.load_data()

    def load_data(self):
        """
        Loads data from pneumonia kaggle dataset
        Binary:
        0: NORMAL
        1: PNEUMONIA

        Multi:
        0: NORMAL
        1: VIRUS PNEUMONIA
        2: BACTERIA PNEUMONIA
        """
        normal_dir = os.path.join(self.root_dir, 'NORMAL')
        pneumonia_dir = os.path.join(self.root_dir, 'PNEUMONIA')

        # Load NORMAL images
        try:
            for img_file in os.listdir(normal_dir):
                self.data.append((os.path.join(normal_dir, img_file), 0))
        except FileNotFoundError as e:
            # sometimes you need '../' instead of './' depending on IDE and PATH
            self.root_dir = '.' + self.root_dir
            normal_dir = os.path.join(self.root_dir, 'NORMAL')
            pneumonia_dir = os.path.join(self.root_dir, 'PNEUMONIA')
            for img_file in os.listdir(normal_dir):
                self.data.append((os.path.join(normal_dir, img_file), 0))

        # Load PNEUMONIA images and classify if required
        for img_file in os.listdir(pneumonia_dir):
            if self.num_classes == 2:
                self.data.append((os.path.join(pneumonia_dir, img_file), 1))
            else:
                if '_virus_' in img_file:
                    self.data.append((os.path.join(pneumonia_dir, img_file), 1))
                elif '_bacteria_' in img_file:
                    self.data.append((os.path.join(pneumonia_dir, img_file), 2))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_loaders(data_dir, batch_size=16, num_workers=3, num_classes=2, image_size=224, pretrained=True):
    train_transform = get_train_transform(image_size=image_size, pretrained=pretrained)
    test_val_transform = get_test_transform(image_size=image_size, pretrained=pretrained)

    train_dataset = ChestXRayDataset(os.path.join(data_dir, 'train'), transform=train_transform, num_classes=num_classes)
    val_dataset = ChestXRayDataset(os.path.join(data_dir, 'val'), transform=test_val_transform, num_classes=num_classes)
    test_dataset = ChestXRayDataset(os.path.join(data_dir, 'test'), transform=test_val_transform, num_classes=num_classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
