import os
import json
import random
import numpy as np
from glob import glob
from PIL import Image
import torch as t
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
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
            img_path, label_info = self.imgs[idx]
        else:
            img_path = self.imgs[idx]
            label_info = None
            
        img = Image.open(img_path)
        original_width, original_height = img.size
        img = np.array(img)

        if self.mode != 'test':
            labels = label_info['label'][:4] + (4 - len(label_info['label'])) * [10]

            bboxes = []
            class_labels = label_info['label']
            for i in range(len(label_info['label'])):
                x_min = label_info['left'][i] / original_width
                y_min = label_info['top'][i] / original_height
                width = label_info['width'][i] / original_width
                height = label_info['height'][i] / original_height
                x_max = x_min + width
                y_max = y_min + height
                bboxes.append([x_min, y_min, x_max, y_max])
            
            if self.aug:
                transform = A.Compose([
                    A.Resize(height=128, width=self.width),
                    A.Affine(
                        scale=(0.9, 1.1),
                        translate_percent=(-0.05, 0.05),
                        rotate=(-15, 15),
                        p=0.5
                    ),
                    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, p=0.5),
                    A.ToGray(p=0.1),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(
                    format='pascal_voc', 
                    label_fields=['class_labels'],
                    min_visibility=0.3,
                    min_area=0.0,
                    check_each_transform=True,
                    clip=True
                ))
            else:
                transform = A.Compose([
                    A.Resize(height=128, width=self.width),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(
                    format='pascal_voc', 
                    label_fields=['class_labels'],
                    min_visibility=0.3,
                    min_area=0.0,
                    check_each_transform=True,
                    clip=True
                ))
            transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
            
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            
            bbox_tensor = t.zeros((4, 4))
            
            for i, bbox in enumerate(transformed_bboxes):
                if i < 4:
                    bbox_tensor[i] = t.tensor(bbox)
            
            return transformed_image, t.tensor(labels).long(), bbox_tensor
        else:
            transform = A.Compose([
                A.Resize(height=128, width=self.width),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            transformed = transform(image=img)
            return transformed['image'], self.imgs[idx]

    def __len__(self):
        return len(self.imgs)

    def collect_fn(self, batch):
        if self.mode != 'test':
            imgs, labels, bboxes = zip(*batch)
            if self.mode == 'train':
                if self.batch_count > 0 and self.batch_count % 10 == 0:
                    self.width = random.choice(range(224, 256, 16))
            self.batch_count += 1
            return t.stack(imgs).float(), t.stack(labels), t.stack(bboxes)
        else:
            imgs, img_paths = zip(*batch)
            self.batch_count += 1
            return t.stack(imgs).float(), img_paths