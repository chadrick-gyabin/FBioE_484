import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import numpy as np

def get_loaders(train_dir, test_dir, batch_size=32, image_size=(512, 512)):
    """
        more stream-lined process for eval.py
    """
    transform = T.Compose([
        T.Resize(image_size),                # Resize the image to the specified size
        T.ToTensor(),                        # Convert the image to a tensor
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
    ])
    
    # Load the training and testing 
    train_ds = ImageFolder(train_dir, transform=transform)
    test_ds  = ImageFolder(test_dir, transform=transform)

    # Create DataLoader objects for both the train and test 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_true_labels(data_loader):
    """
    extracts the truth idx from the data_loader
    """
    true_labels =[]
    for _, labels in data_loader:
        true_labels.append(labels.cpu().detach().numpy())
    return np.concatenate(true_labels, axis=0)  