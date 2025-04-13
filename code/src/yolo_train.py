import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm.auto import tqdm

from config import config
from model import YOLODigitsRNFPN
from dataset import DigitsDataset
from losses import LabelSmoothEntropy
from anchor_utils import YOLOLoss

class YOLOTrainer:
    """
    Trainer class for handling YOLO-based model training and evaluation
    """
    def __init__(self, val=True):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # 初始化数据集和数据加载器
        self.train_set = DigitsDataset(mode='train', aug=True)
        self.train_loader = DataLoader(self.train_set, batch_size=config.batch_size, shuffle=True, num_workers=16,
                                       pin_memory=True, persistent_workers=True,
                                       drop_last=True, collate_fn=self.train_set.collect_fn)
        if val:
            self.val_set = DigitsDataset(mode='val', aug=False)
            self.val_loader = DataLoader(self.val_set, batch_size=config.batch_size,
                                        num_workers=16, pin_memory=True, drop_last=False, 
                                        persistent_workers=True, collate_fn=self.val_set.collect_fn)
        else:
            self.val_loader = None

        # 初始化模型、损失函数、优化器和学习率调度器
        self.model = YOLODigitsRNFPN(class_num=config.class_num).to(self.device)
        self.cls_criterion = LabelSmoothEntropy().to(self.device)
        self.yolo_criterion = YOLOLoss(num_classes=config.class_num).to(self.device)
        
        self.optimizer = Adam(self.model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-08, 
                              weight_decay=config.weights_decay, amsgrad=False)
        self.lr_scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=0)
        self.best_acc = 0
        self.best_checkpoint_path = ""
        
        # 如果有预训练模型则加载
        if config.pretrained is not None:
            self.load_model(config.pretrained)
            if self.val_loader is not None:
                acc = self.eval()
            self.best_acc = acc
            print('Load model from %s, Eval Acc: %.2f' % (config.pretrained, acc * 100))

    def train(self):
        """
        训练模型多个epoch
        """
        for epoch in range(config.start_epoch, config.epoches):
            acc = self.train_epoch(epoch)
            if (epoch + 1) % config.eval_interval == 0:
                print('Start Evaluation')
                if self.val_loader is not None:
                    acc = self.eval()
                # 保存最佳模型
                if acc > self.best_acc:
                    os.makedirs(config.checkpoints, exist_ok=True)
                    save_path = os.path.join(config.checkpoints, 'yolo_best.pth')
                    self.save_model(save_path)
                    print('%s saved successfully...' % save_path)
                    self.best_acc = acc
                    self.best_checkpoint_path = save_path

    def train_epoch(self, epoch):
        """
        训练一个epoch
        
        参数:
            epoch (int): 当前epoch编号
            
        返回:
            float: 当前epoch的准确率
        """
        total_loss = 0
        total_cls_loss = 0
        total_yolo_loss = 0
        corrects = 0
        tbar = tqdm(self.train_loader)
        self.model.train()
        
        for i, (img, label, boxes) in enumerate(tbar):
            img = img.to(self.device)
            label = label.to(self.device)
            
            # 将boxes转换为目标检测格式
            batch_size = img.shape[0]
            target_boxes = []
            target_classes = []
            
            for b in range(batch_size):
                # 收集这个批次样本的所有有效框和类别
                sample_boxes = []
                sample_classes = []
                
                for j in range(4):  # 对于每个数字位置
                    # 忽略空白数字 (类别为10)
                    if label[b, j] != 10:
                        sample_boxes.append(boxes[b, j].to(self.device))
                        sample_classes.append(label[b, j])
                
                if sample_boxes:
                    target_boxes.append(torch.stack(sample_boxes))
                    target_classes.append(torch.tensor(sample_classes, device=self.device))
                else:
                    # 如果没有有效框，添加空列表
                    target_boxes.append(torch.zeros((0, 4), device=self.device))
                    target_classes.append(torch.zeros(0, dtype=torch.long, device=self.device))
            
            # 清空梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            cls_preds, detection_output = self.model(img)
            
            # 如果检测头有输出
            if detection_output is not None:
                # 获取特征图尺寸
                _, _, h, w = detection_output.shape
                
                # 计算stride (原图到特征图的缩放比例)
                stride = img.shape[2] / h  # 或 img.shape[3] / w
                
                # 计算YOLO损失
                targets = list(zip(target_boxes, target_classes))
                yolo_loss = self.yolo_criterion(
                    detection_output,
                    targets, 
                    self.model.anchors,
                    stride,
                    (h, w)
                )
            else:
                yolo_loss = torch.tensor(0.0, device=self.device)
            
            # 计算分类损失
            cls_loss = self.cls_criterion(cls_preds[0], label[:, 0]) + \
                      self.cls_criterion(cls_preds[1], label[:, 1]) + \
                      self.cls_criterion(cls_preds[2], label[:, 2]) + \
                      self.cls_criterion(cls_preds[3], label[:, 3])
            
            # 总损失
            loss = cls_loss + yolo_loss
            loss.backward()
            self.optimizer.step()
            
            # 统计损失
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_yolo_loss += yolo_loss.item()
            
            # 计算准确率
            temp = torch.stack([
                cls_preds[0].argmax(1) == label[:, 0],
                cls_preds[1].argmax(1) == label[:, 1],
                cls_preds[2].argmax(1) == label[:, 2],
                cls_preds[3].argmax(1) == label[:, 3],
            ], dim=1)
            corrects += torch.all(temp, dim=1).sum().item()
            
            # 更新进度条
            tbar.set_description(
                'loss: %.3f (cls: %.3f, yolo: %.3f), acc: %.3f' % 
                (total_loss / (i + 1), 
                 total_cls_loss / (i + 1),
                 total_yolo_loss / (i + 1),
                 corrects * 100 / ((i + 1) * config.batch_size)))
                
            # 学习率调整
            if (i + 1) % config.print_interval == 0:
                self.lr_scheduler.step()
                
        return corrects * 100 / ((i + 1) * config.batch_size)

    def eval(self):
        """
        在验证集上评估模型
        
        返回:
            float: 验证集上的准确率
        """
        self.model.eval()
        corrects = 0
        with torch.no_grad():
            tbar = tqdm(self.val_loader)
            for i, (img, label, boxes) in enumerate(tbar):
                img = img.to(self.device)
                label = label.to(self.device)
                
                # 前向传播，只关注分类结果
                cls_preds, _ = self.model(img)
                
                temp = torch.stack([
                    cls_preds[0].argmax(1) == label[:, 0],
                    cls_preds[1].argmax(1) == label[:, 1],
                    cls_preds[2].argmax(1) == label[:, 2],
                    cls_preds[3].argmax(1) == label[:, 3],
                ], dim=1)
                corrects += torch.all(temp, dim=1).sum().item()
                tbar.set_description('Val Acc: %.2f' % (corrects * 100 / ((i + 1) * config.batch_size)))
        self.model.train()
        return corrects / (len(self.val_loader) * config.batch_size)

    def save_model(self, save_path, save_opt=False, save_config=False):
        """
        保存模型权重，可选择是否保存优化器状态和配置
        
        参数:
            save_path (str): 保存模型的路径
            save_opt (bool): 是否保存优化器状态
            save_config (bool): 是否保存配置
        """
        dicts = {}
        dicts['model'] = self.model.state_dict()
        if save_opt:
            dicts['opt'] = self.optimizer.state_dict()
        if save_config:
            dicts['config'] = {s: config.__getattribute__(s) for s in dir(config) if not s.startswith('_')}
        torch.save(dicts, save_path)

    def load_model(self, load_path, changed=True, save_opt=False, save_config=False):
        """
        加载模型权重，可选择是否加载优化器状态和配置
        
        参数:
            load_path (str): 加载模型的路径
            changed (bool): 模型架构是否有变化
            save_opt (bool): 是否加载优化器状态
            save_config (bool): 是否加载配置
        """
        dicts = torch.load(load_path)
        if not changed:
            self.model.load_state_dict(dicts['model'])
        else:
            # 加载部分权重
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in dicts['model'].items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            
        if save_opt and 'opt' in dicts:
            self.optimizer.load_state_dict(dicts['opt'])
        if save_config and 'config' in dicts:
            for k, v in dicts['config'].items():
                setattr(config, k, v)
