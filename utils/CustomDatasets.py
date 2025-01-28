import os
import tarfile
import urllib

import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt


def plot_images_grid(images, labels, grid_size=(5, 5)):
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    # Flatten the grid of axes for easy iteration
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= len(images):  # Avoid out-of-bound errors
            break
        # Convert image to numpy array if it's a Tensor
        img = images[i].numpy() if isinstance(images[i], torch.Tensor) else images[i]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(labels[i])

    plt.tight_layout()
    plt.show()


class CINIC10(Dataset):
    def __init__(self, data_dir, transform=None, cache_file="cinic10_preprocessed.pt"):
        if os.path.exists(cache_file):
            # Load preprocessed data
            print(f"Loading dataset from {cache_file}...")
            self.data, self.targets = torch.load(cache_file)
        else:
            # Preprocess and save dataset
            print(f"Preprocessing and saving dataset to {cache_file}...")

            # Load ImageFolder dataset to access image paths and labels
            self.image_folder = datasets.ImageFolder(root=data_dir)

            # Extract all image paths and corresponding labels
            self.data = []
            self.targets = []
            imgs_and_labels = self.image_folder.imgs

            random.shuffle(imgs_and_labels)

            # Load all images into memory and store them as tensors
            for img_path, label in imgs_and_labels:
                image = Image.open(img_path).convert('RGB')  # Ensure images are RGB
                self.data.append(np.array(image))  # Store the raw image as numpy array
                self.targets.append(label)  # Store the label

            # Convert lists to tensors
            self.data = torch.tensor(np.array(self.data))  # Convert images to a tensor
            self.targets = torch.tensor(self.targets)  # Convert labels to a tensor
            torch.save((self.data, self.targets), cache_file)

            # plot_images_grid(self.data[:25], self.targets[:25])

        # Store the transform
        self.transform = transform

    def __len__(self):
        # Return the total number of images
        return len(self.data)

    def __getitem__(self, idx):
        # Get image and label
        image, label = self.data[idx], self.targets[idx]

        # Convert image to PIL Image if transforms are required
        image = Image.fromarray(image.numpy())  # Convert back to PIL Image

        # Apply the transforms, if any
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def download_CINIC10(data_path):
    cinic_directory = data_path + 'CINIC10'
    try:
        if not os.path.exists(cinic_directory):
            filename = './CINIC-10.tar.gz'
            if not os.path.exists(filename):
                url = 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz'
                print("Downloading CINIC-10 Dataset")
                urllib.request.urlretrieve(url, filename)
                print("Download complete!")

            os.mkdir(cinic_directory)
            with tarfile.open(filename, "r:gz") as tar:
                tar.extractall(path=cinic_directory)
            print(f"Extraction of CINIC-10 Dataset Complete!")
    except Exception as e:
        print(f"Error occurred: {e}")


def load_CINIC10(data_path):
    download_CINIC10(data_path)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    training_dataset = CINIC10(data_dir=data_path + 'CINIC10/train', transform=transform, cache_file=data_path + 'CINIC10/cinic10_preprocessed_train.pt')
    test_dataset = CINIC10(data_dir=data_path + 'CINIC10/test', transform=transform, cache_file=data_path + 'CINIC10/cinic10_preprocessed_test.pt')

    return training_dataset, test_dataset, transform
