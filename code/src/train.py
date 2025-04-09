import os
import torch as t
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm.auto import tqdm

from config import config
from model import LabelSmoothEntropy, DigitsResnet152
from dataset import DigitsDataset

class Trainer:
    """
    Trainer class for handling model training and evaluation
    """
    def __init__(self, val=True):
        self.device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
        # Init datasets and dataloaders
        self.train_set = DigitsDataset(mode='train', aug=True)
        self.train_loader = DataLoader(self.train_set, batch_size=config.batch_size, shuffle=True, num_workers=16,
                                       pin_memory=True, persistent_workers=True,
                                       drop_last=True, collate_fn=self.train_set.collect_fn)
        if val:
            self.val_set = DigitsDataset(mode='val', aug=False)
            self.val_loader = DataLoader(self.val_set, batch_size=config.batch_size,
                                        num_workers=16, pin_memory=True, drop_last=False, 
                                        persistent_workers=True)
        else:
            self.val_loader = None

        # Init model, criterion, optimizer and scheduler
        self.model = DigitsResnet152(class_num=config.class_num).to(self.device)
        self.criterion = LabelSmoothEntropy().to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                              amsgrad=False)
        self.lr_scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=0)
        self.best_acc = 0
        self.best_checkpoint_path = ""
        
        # Load pretrained model if available
        if config.pretrained is not None:
            self.load_model(config.pretrained)
            if self.val_loader is not None:
                acc = self.eval()
            self.best_acc = acc
            print('Load model from %s, Eval Acc: %.2f' % (config.pretrained, acc * 100))

    def train(self):
        """
        Train the model for multiple epochs
        """
        for epoch in range(config.start_epoch, config.epoches):
            acc = self.train_epoch(epoch)
            if (epoch + 1) % config.eval_interval == 0:
                print('Start Evaluation')
                if self.val_loader is not None:
                    acc = self.eval()
                # Save best model
                if acc > self.best_acc:
                    os.makedirs(config.checkpoints, exist_ok=True)
                    save_path = os.path.join(config.checkpoints, 'best.pth')
                    self.save_model(save_path)
                    print('%s saved successfully...' % save_path)
                    self.best_acc = acc
                    self.best_checkpoint_path = save_path

    def train_epoch(self, epoch):
        """
        Train the model for one epoch
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            float: Accuracy for this epoch
        """
        total_loss = 0
        corrects = 0
        tbar = tqdm(self.train_loader)
        self.model.train()
        for i, (img, label) in enumerate(tbar):
            img = img.to(self.device)
            label = label.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(img)
            # Calculate loss for all 4 digits
            loss = self.criterion(pred[0], label[:, 0]) + \
                   self.criterion(pred[1], label[:, 1]) + \
                   self.criterion(pred[2], label[:, 2]) + \
                   self.criterion(pred[3], label[:, 3])
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            # Calculate accuracy (all 4 digits correct)
            temp = t.stack([ \
                pred[0].argmax(1) == label[:, 0], \
                pred[1].argmax(1) == label[:, 1], \
                pred[2].argmax(1) == label[:, 2], \
                pred[3].argmax(1) == label[:, 3], ], dim=1)
            corrects += t.all(temp, dim=1).sum().item()
            tbar.set_description(
                'loss: %.3f, acc: %.3f' % (loss / (i + 1), corrects * 100 / ((i + 1) * config.batch_size)))
            if (i + 1) % config.print_interval == 0:
                self.lr_scheduler.step()
        return corrects * 100 / ((i + 1) * config.batch_size)

    def eval(self):
        """
        Evaluate the model on validation dataset
        
        Returns:
            float: Accuracy on validation dataset
        """
        self.model.eval()
        corrects = 0
        with t.no_grad():
            tbar = tqdm(self.val_loader)
            for i, (img, label) in enumerate(tbar):
                img = img.to(self.device)
                label = label.to(self.device)
                pred = self.model(img)
                temp = t.stack([
                    pred[0].argmax(1) == label[:, 0], \
                    pred[1].argmax(1) == label[:, 1], \
                    pred[2].argmax(1) == label[:, 2], \
                    pred[3].argmax(1) == label[:, 3], \
                    ], dim=1)
                corrects += t.all(temp, dim=1).sum().item()
                tbar.set_description('Val Acc: %.2f' % (corrects * 100 / ((i + 1) * config.batch_size)))
        self.model.train()
        return corrects / (len(self.val_loader) * config.batch_size)

    def save_model(self, save_path, save_opt=False, save_config=False):
        """
        Save model weights and optionally optimizer state and config
        
        Args:
            save_path (str): Path to save the model
            save_opt (bool): Whether to save optimizer state
            save_config (bool): Whether to save config
        """
        dicts = {}
        dicts['model'] = self.model.state_dict()
        if save_opt:
            dicts['opt'] = self.optimizer.state_dict()
        if save_config:
            dicts['config'] = {s: config.__getattribute__(s) for s in dir(config) if not s.startswith('_')}
        t.save(dicts, save_path)

    def load_model(self, load_path, changed=False, save_opt=False, save_config=False):
        """
        Load model weights and optionally optimizer state and config
        
        Args:
            load_path (str): Path to load the model from
            changed (bool): Whether the model architecture has changed
            save_opt (bool): Whether to load optimizer state
            save_config (bool): Whether to load config
        """
        dicts = t.load(load_path)
        if not changed:
            self.model.load_state_dict(dicts['model'])

        if save_opt:
            self.optimizer.load_state_dict(dicts['opt'])

        if save_config:
            for k, v in dicts['config'].items():
                config.__setattr__(k, v)