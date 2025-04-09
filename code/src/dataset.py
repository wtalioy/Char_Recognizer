import os
import json
import random
from glob import glob
from PIL import Image
import torch as t
from torch.utils.data import Dataset
from torchvision import transforms
from config import data_dir, transform

class DigitsDataset(Dataset):
    """
    DigitsDataset for character recognition task
    
    Params:
      mode (str): 'train', 'val' or 'test', default: 'train'
      size (tuple): image size, default: (128, 256)
      aug (bool): whether to use image augmentation, default: True
    """
    def __init__(self, mode='train'):
        super(DigitsDataset, self).__init__()
        self.mode = mode
        self.width = 224
        self.batch_count = 0
        self.transform = transform[mode]
        if mode == 'test':
            self.imgs = glob(data_dir['test_data'] + '*.png')
            self.labels = None
        else:
            labels = json.load(open(data_dir['%s_label' % mode], 'r'))
            imgs = glob(data_dir['%s_data' % mode] + '*.png')
            self.imgs = [(img, labels[os.path.split(img)[-1]]) for img in imgs \
                         if os.path.split(img)[-1] in labels]

    def __getitem__(self, idx):
        if self.mode != 'test':
            img, label = self.imgs[idx]
        else:
            img = self.imgs[idx]
            label = None
        img = Image.open(img)
        trans0 = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        if self.mode != 'test':
            return self.transform(img), t.tensor(
                label['label'][:4] + (4 - len(label['label'])) * [10]).long()
        else:
            # trans1.append(transforms.RandomErasing(scale=(0.02, 0.1)))
            return self.transform(img), self.imgs[idx]

    def __len__(self):
        return len(self.imgs)

    def collect_fn(self, batch):
        imgs, labels = zip(*batch)
        if self.mode == 'train':
            if self.batch_count > 0 and self.batch_count % 10 == 0:
                self.width = random.choice(range(224, 256, 16))

        self.batch_count += 1
        return t.stack(imgs).float(), t.stack(labels)