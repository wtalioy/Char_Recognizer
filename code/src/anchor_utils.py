import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def generate_anchors(anchor_sizes, device='cpu'):
    """
    生成用于模型的anchors
    
    参数:
    - anchor_sizes: list，每个特征图层的anchor尺寸列表，如 [[w1, h1], [w2, h2], [w3, h3]]
    - device: 设备
    
    返回:
    - 张量形状 (num_anchors, 2)，表示anchors的宽高
    """
    anchors = torch.tensor(anchor_sizes, dtype=torch.float32, device=device)
    return anchors

def decode_yolo(pred, anchors, stride, grid_size):
    """
    解码YOLO输出为边界框坐标
    
    参数:
    - pred: 预测输出, shape (batch_size, num_anchors*(5+num_classes), h, w)
    - anchors: 锚框, shape (num_anchors, 2)
    - stride: 特征图相对于原图的步长
    - grid_size: 特征图大小 (h, w)
    
    返回:
    - boxes: 解码后的边界框, shape (batch_size, num_anchors, h, w, 4) - 格式为(x1, y1, x2, y2)
    - objectness: 置信度, shape (batch_size, num_anchors, h, w)
    - class_scores: 类别得分, shape (batch_size, num_anchors, h, w, num_classes)
    """
    batch_size = pred.shape[0]
    num_anchors = len(anchors)
    h, w = grid_size
    num_classes = pred.shape[1] // num_anchors - 5
    
    # 重塑预测为 (batch, num_anchors, 5+num_classes, h, w)，然后转置为 (batch, num_anchors, h, w, 5+num_classes)
    pred = pred.view(batch_size, num_anchors, 5 + num_classes, h, w).permute(0, 1, 3, 4, 2).contiguous()
    
    # 解析预测值
    tx = torch.sigmoid(pred[..., 0])  # center x offset
    ty = torch.sigmoid(pred[..., 1])  # center y offset
    tw = pred[..., 2]  # width scale
    th = pred[..., 3]  # height scale
    objectness = torch.sigmoid(pred[..., 4])  # objectness score
    class_scores = torch.sigmoid(pred[..., 5:])  # class probabilities
    
    # 生成网格坐标 (h, w, 2)
    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=pred.device), 
                                   torch.arange(w, device=pred.device), 
                                   indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=-1).float()
    
    # 扩展网格维度以匹配预测 (1, 1, h, w, 2)
    grid = grid.view(1, 1, h, w, 2)
    
    # 扩展anchors维度 (1, num_anchors, 1, 1, 2)
    anchors = anchors.view(1, num_anchors, 1, 1, 2)
    
    # 计算中心点坐标
    xy = (torch.stack([tx, ty], dim=-1) + grid) * stride
    
    # 计算宽高
    wh = torch.exp(torch.stack([tw, th], dim=-1)) * anchors
    
    # 转换为左上右下格式 (x1, y1, x2, y2)
    x1y1 = xy - wh / 2
    x2y2 = xy + wh / 2
    boxes = torch.cat([x1y1, x2y2], dim=-1)
    
    return boxes, objectness, class_scores

class YOLOLoss(nn.Module):
    """
    YOLO损失函数
    """
    def __init__(self, num_classes=11, num_anchors=3, obj_weight=1.0, noobj_weight=0.5, coord_weight=5.0, cls_weight=1.0):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # 损失权重
        self.obj_weight = obj_weight  # 有目标的权重
        self.noobj_weight = noobj_weight  # 无目标的权重
        self.coord_weight = coord_weight  # 坐标损失的权重
        self.cls_weight = cls_weight  # 分类损失的权重
        
        # 使用BCE损失用于分类和置信度
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.ciou_loss = None  # 将在forward中初始化，以支持动态设备
        
        self.ignore_thres = 0.5  # IoU阈值，用于判断是否忽略预测
        
    def forward(self, predictions, targets, anchors, stride, grid_size):
        """
        计算YOLO损失
        
        参数:
        - predictions: 模型原始输出 (batch_size, num_anchors*(5+num_classes), h, w)
        - targets: 格式为 [(boxes, classes), ...] 的列表，每个元素对应一个批次图像
                   boxes: 目标框 (num_boxes, 4) - 格式为(x1, y1, x2, y2)
                   classes: 目标类别 (num_boxes,)
        - anchors: anchor尺寸 (num_anchors, 2)
        - stride: 特征图相对于原图的步长
        - grid_size: 特征图大小 (h, w)
        
        返回:
        - 总损失值
        """
        batch_size = predictions.shape[0]
        h, w = grid_size
        device = predictions.device
        
        # 初始化CIoU损失函数（在这里初始化以支持动态设备）
        if self.ciou_loss is None or self.ciou_loss.eps != 1e-7:  # 防止多次初始化
            from losses import CIoULoss
            self.ciou_loss = CIoULoss().to(device)
        
        # 重塑预测为 (batch, num_anchors, 5+num_classes, h, w)，然后转置为 (batch, num_anchors, h, w, 5+num_classes)
        pred = predictions.view(batch_size, self.num_anchors, 5 + self.num_classes, h, w).permute(0, 1, 3, 4, 2).contiguous()
        
        # 提取预测的各个部分
        pred_tx = pred[..., 0]  # center x 
        pred_ty = pred[..., 1]  # center y
        pred_tw = pred[..., 2]  # width
        pred_th = pred[..., 3]  # height
        pred_obj = pred[..., 4]  # objectness
        pred_cls = pred[..., 5:]  # class predictions
        
        # 生成网格坐标 
        grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), 
                                        torch.arange(w, device=device), 
                                        indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1).float()
        
        # 扩展网格维度以匹配预测 (1, 1, h, w, 2)
        grid = grid.view(1, 1, h, w, 2)
        
        # 扩展anchors维度 (1, num_anchors, 1, 1, 2)
        anchors = anchors.view(1, self.num_anchors, 1, 1, 2)
        
        # 初始化统计损失和mask
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)
        loss_cls = torch.zeros(1, device=device)
        
        # 创建objectness掩码，初始化为没有目标
        obj_mask = torch.zeros_like(pred_obj, dtype=torch.bool)
        noobj_mask = torch.ones_like(pred_obj, dtype=torch.bool)
        
        num_targets = 0  # 跟踪目标的总数
        
        # 处理每个批次的图像
        for batch_idx in range(batch_size):
            target_boxes, target_classes = targets[batch_idx]
            
            if len(target_boxes) == 0:
                continue
                
            num_targets += len(target_boxes)
            
            # 将目标框转换为中心点宽高格式 (xcenter, ycenter, w, h)，并归一化到特征图尺度
            target_boxes_xywh = self._xyxy_to_xywh(target_boxes)
            
            # 为每个目标找到最佳的anchor
            for target_idx, (box_xywh, cls_id) in enumerate(zip(target_boxes_xywh, target_classes)):
                # 归一化目标坐标到网格尺度
                gx, gy, gw, gh = box_xywh
                gx = gx / stride
                gy = gy / stride
                gw = gw / stride
                gh = gh / stride
                
                # 获取网格索引
                gi = int(gx)
                gj = int(gy)
                
                # 确保网格索引在合法范围内
                if gi >= w or gj >= h or gi < 0 or gj < 0:
                    continue
                
                # 计算与每个anchor的IoU，选择最佳匹配
                target_box_tensor = torch.tensor([[0, 0, gw, gh]], device=device)
                anchor_shapes = torch.cat((torch.zeros(self.num_anchors, 2, device=device), 
                                          anchors.view(-1, 2)), 1)
                
                # 计算target与每个anchor的宽高比
                wh_ratio = target_box_tensor[:, 2:4] / anchor_shapes[:, 2:4]
                ratio_penalty = torch.max(wh_ratio, 1/wh_ratio).max(1)[0]
                
                # 选择比率惩罚最小的anchor
                best_anchor_idx = torch.argmin(ratio_penalty)
                
                # 设置正样本掩码
                obj_mask[batch_idx, best_anchor_idx, gj, gi] = True
                noobj_mask[batch_idx, best_anchor_idx, gj, gi] = False
                
                # 计算目标tx, ty (相对于网格点的偏移)
                tx = gx - gi
                ty = gy - gj
                
                # 计算目标tw, th (相对于anchor的缩放)
                tw = torch.log(gw / anchors[0, best_anchor_idx, 0, 0, 0] + 1e-16)
                th = torch.log(gh / anchors[0, best_anchor_idx, 0, 0, 1] + 1e-16)
                
                # 计算坐标损失 (MSE)
                loss_x = self.mse_loss(pred_tx[batch_idx, best_anchor_idx, gj, gi], tx)
                loss_y = self.mse_loss(pred_ty[batch_idx, best_anchor_idx, gj, gi], ty)
                loss_w = self.mse_loss(pred_tw[batch_idx, best_anchor_idx, gj, gi], tw)
                loss_h = self.mse_loss(pred_th[batch_idx, best_anchor_idx, gj, gi], th)
                
                # 计算CIoU损失 (也可以用MSE损失替代)
                # 将预测解码为实际坐标
                pred_x = torch.sigmoid(pred_tx[batch_idx, best_anchor_idx, gj, gi]) + gi
                pred_y = torch.sigmoid(pred_ty[batch_idx, best_anchor_idx, gj, gi]) + gj
                pred_w = torch.exp(pred_tw[batch_idx, best_anchor_idx, gj, gi]) * anchors[0, best_anchor_idx, 0, 0, 0]
                pred_h = torch.exp(pred_th[batch_idx, best_anchor_idx, gj, gi]) * anchors[0, best_anchor_idx, 0, 0, 1]
                
                # 转换为xyxy格式
                pred_x1 = (pred_x - pred_w / 2) * stride
                pred_y1 = (pred_y - pred_h / 2) * stride
                pred_x2 = (pred_x + pred_w / 2) * stride
                pred_y2 = (pred_y + pred_h / 2) * stride
                
                pred_box = torch.tensor([[pred_x1, pred_y1, pred_x2, pred_y2]], device=device)
                target_box = target_boxes[target_idx:target_idx+1]  # 使用原始xyxy格式的目标框
                
                # 使用CIoU损失
                box_loss = self.ciou_loss(pred_box, target_box)
                
                # 计算类别损失
                target_cls = torch.zeros(self.num_classes, device=device)
                target_cls[cls_id] = 1
                cls_loss = self.bce_loss(pred_cls[batch_idx, best_anchor_idx, gj, gi], target_cls).sum()
                
                # 累加损失
                loss_box += box_loss
                loss_cls += cls_loss
            
            # 设置objectness目标（有目标的为1，无目标的为0）
            target_obj = torch.zeros_like(pred_obj, device=device)
            target_obj[obj_mask] = 1
            
            # 计算objectness损失
            loss_obj += self.bce_loss(pred_obj, target_obj).mean()
        
        # 如果没有目标，返回仅包含背景的损失
        if num_targets == 0:
            loss_obj = self.bce_loss(pred_obj, torch.zeros_like(pred_obj)).mean()
            return loss_obj
        
        # 应用权重并求平均
        loss_box = self.coord_weight * loss_box / num_targets
        loss_obj = self.obj_weight * loss_obj / batch_size
        loss_cls = self.cls_weight * loss_cls / num_targets
        
        # 计算总损失
        total_loss = loss_box + loss_obj + loss_cls
        
        return total_loss
        
    def _xyxy_to_xywh(self, boxes):        
        """
        将 (x1, y1, x2, y2) 格式转换为 (xcenter, ycenter, w, h) 格式
        """
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        w = x2 - x1
        h = y2 - y1
        xcenter = x1 + w/2
        ycenter = y1 + h/2
        return torch.stack([xcenter, ycenter, w, h], dim=-1)
