import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision import transforms


class LFWDataset(Dataset):
    def __init__(self, data_dir, img_size=100):
        self.data_dir = data_dir
        self.image_folder = os.path.join(self.data_dir, 'lfw')
        self.person_to_images_map = self._load_image_paths()

        # 创建transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()])

    def _load_image_paths(self):
        person_to_images_map = {}
        for person_name in os.listdir(self.image_folder):
            person_dir = os.path.join(self.image_folder, person_name)
            if os.path.isdir(person_dir):
                for image_name in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_name)
                    if person_name not in person_to_images_map:
                        person_to_images_map[person_name] = []
                    person_to_images_map[person_name].append(image_path)
        return person_to_images_map

    def __len__(self):
        return sum([len(images) for images in self.person_to_images_map.values()]) * 2

    def __getitem__(self, idx):
        person1, person2 = np.random.choice(list(self.person_to_images_map.keys()), 2, replace=False)
        image1 = np.random.choice(self.person_to_images_map[person1])

        label = None
        if idx % 2 == 0:  # same person
            image2 = np.random.choice(self.person_to_images_map[person1])
            label = 1.0
        else:  # different person
            image2 = np.random.choice(self.person_to_images_map[person2])
            label = 0.0

        image1 = Image.open(image1)
        image2 = Image.open(image2)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))
