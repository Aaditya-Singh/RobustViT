import json
from torch.utils import data
from torchvision.datasets import ImageFolder
import torch
import os
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
from munkres import Munkres
import multiprocessing
from multiprocessing import Process, Manager
import collections
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import torchvision
import cv2
from label_str_to_imagenet_classes import label_str_to_imagenet_classes

torch.manual_seed(0)

ImageItem = collections.namedtuple('ImageItem', ('image_name', 'tag'))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

class RobustnessDataset(ImageFolder):
    def __init__(self, imagenet_path, imagenet_classes_path='imagenet_classes.json', isV2=False, isSI=False):
        self._isV2 = isV2
        self._isSI = isSI
        self._imagenet_path = imagenet_path
        with open(imagenet_classes_path, 'r') as f:
            self._imagenet_classes = json.load(f)
        self._tag_list = [tag for tag in os.listdir(self._imagenet_path)]
        self._all_images = []
        for tag in self._tag_list:
            base_dir = os.path.join(self._imagenet_path, tag)
            for i, file in enumerate(os.listdir(base_dir)):
                self._all_images.append(ImageItem(file, tag))


    def __getitem__(self, item):
        image_item = self._all_images[item]
        image_path = os.path.join(self._imagenet_path, image_item.tag, image_item.image_name)
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = transform(image)

        if self._isV2:
            class_name = int(image_item.tag)
        elif self._isSI:
            class_name = int(label_str_to_imagenet_classes[image_item.tag])
        else:
            class_name = int(self._imagenet_classes[image_item.tag])

        return image, class_name

    def __len__(self):
        return len(self._all_images)