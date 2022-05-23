import os
import json
import re

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption


class roco_train(Dataset):
    def __init__(self, transform, dataset_root, max_words=30, prompt=''):
        """
        dataset_root (string): Root directory of dataset (e.g. dataset/roco)
        """
        filename = 'train_caption.json'


        self.annotation = json.load(open(os.path.join(dataset_root, filename), 'r'))
        self.transform = transform
        self.image_root = dataset_root
        self.max_words = max_words
        self.prompt = prompt

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = self.prompt + pre_caption(ann['caption'], self.max_words)

        return image, caption, self.img_ids[ann['image_id']]


class roco_eval(Dataset):
    def __init__(self, transform, dataset_root, split):
        """
        dataset_root (string): Root directory of dataset (e.g. dataset/roco)
        split (string): val or test
        """
        filenames = {'val': 'val_caption.json', 'test': 'test_caption.json'}

        self.annotation = json.load(open(os.path.join(dataset_root, filenames[split]), 'r'))
        self.transform = transform
        self.image_root = dataset_root

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        img_id = ann['image_id']

        return image, int(img_id)

