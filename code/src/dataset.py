import os
import json
import random
from glob import glob
from PIL import Image
import torch as t
from torch.utils.data import Dataset
from torchvision import transforms
from config import data_dir

class DigitsDataset(Dataset):
    """
    DigitsDataset
    Params:
      data_dir(string): data directory
      label_path(string): label path
      aug(bool): wheather do image augmentation, default: True
    """
    def __init__(self, mode='train', size=(128, 256), aug=True):
        super(DigitsDataset, self).__init__()
        self.aug = aug
        self.size = size
        self.mode = mode
        self.width = 224
        self.batch_count = 0
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
        min_size = self.size[0] if (img.size[1] / self.size[0]) < ((img.size[0] / self.size[1])) else self.size[1]
        trans1 = [
            transforms.Resize(128),
            transforms.CenterCrop((128, self.width))
        ]
        if self.aug:
            trans1.extend([
                transforms.ColorJitter(0.1, 0.1, 0.1),
                transforms.RandomGrayscale(0.1),
                transforms.RandomAffine(15, translate=(0.05, 0.1), shear=5)
            ])
        trans1.extend(trans0)
        if self.mode != 'test':
            return transforms.Compose(trans1)(img), t.tensor(
                label['label'][:4] + (4 - len(label['label'])) * [10]).long()
        else:
            # trans1.append(transforms.RandomErasing(scale=(0.02, 0.1)))
            return transforms.Compose(trans1)(img), self.imgs[idx]

    def __len__(self):
        return len(self.imgs)

    def collect_fn(self, batch):
        imgs, labels = zip(*batch)
        if self.mode == 'train':
            if self.batch_count > 0 and self.batch_count % 10 == 0:
                self.width = random.choice(range(224, 256, 16))

        self.batch_count += 1
        return t.stack(imgs).float(), t.stack(labels) 