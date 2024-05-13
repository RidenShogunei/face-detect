import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision import transforms
import cv2

def crop_face(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # img现在直接就是一个numpy array，不用再使用cv2.imread()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = img[y:y+h, x:x+w]
    return img

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
        image1_path = np.random.choice(self.person_to_images_map[person1])

        if idx % 2 == 0:  # same person
            image2_path = np.random.choice(self.person_to_images_map[person1])
        else:  # different person
            image2_path = np.random.choice(self.person_to_images_map[person2])

        image1 = Image.open(image1_path)
        image1 = np.array(image1)
        image1 = image1[:, :, ::-1].copy()  # Convert RGB to BGR
        image1 = crop_face(image1)
        image1 = Image.fromarray(image1)
        image2 = Image.open(image2_path)
        image2 = np.array(image2)
        image2 = image2[:, :, ::-1].copy()  # Convert RGB to BGR
        image2 = crop_face(image2)
        image2 = Image.fromarray(image2)
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        # flatten the tensors and calculate the euclidean distance
        euclidean_distance = torch.dist(image1.view(-1), image2.view(-1), p=2)
        return image1, image2, euclidean_distance
