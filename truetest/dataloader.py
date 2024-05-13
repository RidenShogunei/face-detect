import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random

class LFWDataset(Dataset):
    def __init__(self, data_dir, same_rate=0.7, transform=None):
        self.transform = transform
        self.data_dir = data_dir
        self.same_rate = same_rate
        self.image_folder = os.path.join(self.data_dir, 'lfw')
        self.image_paths = self._load_image_paths()

    def _load_image_paths(self):
        image_paths = {}
        for label, person_name in enumerate(os.listdir(self.image_folder)):
            person_dir = os.path.join(self.image_folder, person_name)
            if os.path.isdir(person_dir):
                image_paths[label] = []
                for image_name in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_name)
                    image_paths[label].append(image_path)
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_labels = list(self.image_paths.keys())
        same_person = torch.rand(1) < self.same_rate
        label = image_labels[idx]
        images_of_label = self.image_paths[label]

        first_image_path = random.choice(images_of_label)
        first_image = Image.open(first_image_path)

        if self.transform:
            first_image = self.transform(first_image)

        if same_person:
            # pick second image from same person
            images_of_label.remove(first_image_path)
            second_image_path = random.choice(images_of_label if images_of_label else self.image_paths[random.choice([lbl for lbl in image_labels if lbl != label])])
        else:
            # pick second image from different person
            second_image_path = random.choice(self.image_paths[random.choice([lbl for lbl in image_labels if lbl != label])])

        second_image = Image.open(second_image_path)

        if self.transform:
            second_image = self.transform(second_image)
        return first_image, second_image, same_person.float()

def get_loader(data_dir, batch_size, same_rate=0.7):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = LFWDataset(data_dir, same_rate, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader