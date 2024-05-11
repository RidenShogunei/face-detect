import os
import pandas as pd
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class LFWDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.data_dir = data_dir
        self.image_folder = os.path.join(self.data_dir, 'lfw')

        self.image_paths, self.labels = self._load_image_paths()

    def _load_image_paths(self):
        image_paths = []
        labels = []
        for label, person_name in enumerate(os.listdir(self.image_folder)):
            person_dir = os.path.join(self.image_folder, person_name)
            if os.path.isdir(person_dir):
                for image_name in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_name)
                    image_paths.append(image_path)
                    labels.append(label)

        print(labels)
        return image_paths, labels


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_loader(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = LFWDataset(data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader