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

torch.manual_seed(0)

SegItem = collections.namedtuple('SegItem', ('image_name', 'tag'))
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

TRANSFORM_TRAIN = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
])

TRANSFORM_EVAL = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

IMAGE_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

MERGED_TAGS = {'n04356056', 'n04355933',
               'n04493381', 'n02808440',
               'n03642806', 'n03832673',
               'n04008634', 'n03773504',
               'n03887697', 'n15075141'}

IN_CLASSES_PATH = "imagenet_classes.json"

class SegmentationDataset(ImageFolder):
    def __init__(self, seg_path, imagenet_path, imagenet_classes_path=IN_CLASSES_PATH):
        self._seg_path = seg_path
        self._imagenet_path = imagenet_path
        with open(imagenet_classes_path, 'r') as f:
            self._imagenet_classes = json.load(f)
        # self._tag_list = [rel_path.split('_')[0] for rel_path in os.listdir(self._seg_path) \
        #     if rel_path.split('_')[0] not in MERGED_TAGS]
        self._tag_list = [rel_path.split('_')[0] for rel_path in os.listdir(self._seg_path)]
        for tag in self._tag_list:
            assert tag in self._imagenet_classes
        self._all_segementations = []
        for rel_path in os.listdir(self._seg_path):
            tag = rel_path.split('_')[0]
            # if tag in MERGED_TAGS: continue
            seg_name = rel_path.split('.')[0]
            if ('bfs' not in seg_name) or ('.JPEG' in rel_path): continue
            seg_path = os.path.join(self._seg_path, rel_path)
            # seg_map = torch.load(seg_path)
            # seg_map = torch.from_numpy(seg_map.astype(np.float32))
            # if torch.sum(seg_map) < 520: continue
            self._all_segementations.append(SegItem(seg_name, tag))

    def __getitem__(self, item):
        seg_item = self._all_segementations[item]

        seg_path = os.path.join(self._seg_path, seg_item.image_name + ".pt")

        image_path = os.path.join(self._imagenet_path, seg_item.tag, seg_item.image_name.split('_tokencut_bfs')[0] + ".JPEG")
        image = Image.open(image_path)
        image = image.convert('RGB')

        seg_map = torch.load(seg_path)
        seg_map = torch.from_numpy(seg_map.astype(np.float32))

        # transforms - start
        seg_map = seg_map.reshape(1, seg_map.shape[-2], seg_map.shape[-1])
        resize = transforms.Resize(size=(256, 256))
        image = resize(image)
        seg_map = resize(seg_map)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(224, 224))
        image = TF.crop(image, i, j, h, w)
        seg_map = TF.crop(seg_map, i, j, h, w)

        # RandomHorizontalFlip
        if random.random() > 0.5:
            image = TF.hflip(image)
            seg_map = TF.hflip(seg_map)

        image_ten = IMAGE_TRANSFORMS(image)
        # transforms - end

        class_name = int(self._imagenet_classes[seg_item.tag])

        return seg_map, image_ten, class_name

    def __len__(self):
        return len(self._all_segementations)
