import torch
import torch.nn as nn
import math

class LabelSmoothEntropy(nn.Module):
    """
    Label smoothing entropy loss for more robust training
    """
    def __init__(self, smooth=0.1, class_weights=None, size_average='mean'):
        super(LabelSmoothEntropy, self).__init__()
        self.size_average = size_average
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, preds, targets):
        lb_pos, lb_neg = 1 - self.smooth, self.smooth / (preds.shape[0] - 1)
        smoothed_lb = torch.zeros_like(preds).fill_(lb_neg).scatter_(1, targets[:, None], lb_pos)
        log_soft = torch.nn.functional.log_softmax(preds, dim=1)
        if self.class_weights is not None:
            loss = -log_soft * smoothed_lb * self.class_weights[None, :]
        else:
            loss = -log_soft * smoothed_lb
        loss = loss.sum(1)
        if self.size_average == 'mean':
            return loss.mean()
        elif self.size_average == 'sum':
            return loss.sum()
        else:
            raise NotImplementedError


class CIoULoss(nn.Module):
    """
    Complete IoU Loss
    包含IoU、中心点距离、长宽比在内的完整IoU损失
    
    修正版本，避免数值不稳定性导致NaN
    """
    def __init__(self, eps=1e-7):
        super(CIoULoss, self).__init__()
        self.eps = eps
        
    def forward(self, preds, targets):
        """
        参数:
            preds: 预测边界框 [x_min, y_min, x_max, y_max], 形状为 [batch_size, 4]
            targets: 目标边界框 [x_min, y_min, x_max, y_max], 形状为 [batch_size, 4]
        返回:
            ciou_loss: CIoU损失值
        """
        # 确保输入形状正确
        if preds.shape[0] == 0 or targets.shape[0] == 0:
            return torch.tensor(0.0, device=preds.device)
            
        # 预测和目标边界框的宽高（移除+1）
        w_pred = (preds[:, 2] - preds[:, 0]).clamp(min=self.eps)
        h_pred = (preds[:, 3] - preds[:, 1]).clamp(min=self.eps)
        w_target = (targets[:, 2] - targets[:, 0]).clamp(min=self.eps)
        h_target = (targets[:, 3] - targets[:, 1]).clamp(min=self.eps)
        
        # 预测和目标边界框的面积
        area_pred = w_pred * h_pred
        area_target = w_target * h_target
        
        # 计算IoU
        # 交集坐标
        left = torch.max(preds[:, 0], targets[:, 0])
        top = torch.max(preds[:, 1], targets[:, 1])
        right = torch.min(preds[:, 2], targets[:, 2])
        bottom = torch.min(preds[:, 3], targets[:, 3])
        
        # 交集宽高和面积
        w_intersect = (right - left).clamp(min=0)
        h_intersect = (bottom - top).clamp(min=0)
        area_intersect = w_intersect * h_intersect
        
        # 并集面积
        area_union = area_pred + area_target - area_intersect
        
        # IoU
        iou = area_intersect / (area_union + self.eps)
        
        # 外接矩形坐标
        left_enclosing = torch.min(preds[:, 0], targets[:, 0])
        top_enclosing = torch.min(preds[:, 1], targets[:, 1])
        right_enclosing = torch.max(preds[:, 2], targets[:, 2])
        bottom_enclosing = torch.max(preds[:, 3], targets[:, 3])
        
        # 外接矩形对角线长度的平方（修正计算）
        c_w = (right_enclosing - left_enclosing).clamp(min=self.eps)
        c_h = (bottom_enclosing - top_enclosing).clamp(min=self.eps)
        c2 = c_w**2 + c_h**2 + self.eps
        
        # 预测和目标边界框中心点
        center_x_pred = (preds[:, 0] + preds[:, 2]) / 2
        center_y_pred = (preds[:, 1] + preds[:, 3]) / 2
        center_x_target = (targets[:, 0] + targets[:, 2]) / 2
        center_y_target = (targets[:, 1] + targets[:, 3]) / 2
        
        # 中心点距离的平方
        center_dist2 = (center_x_pred - center_x_target)**2 + (center_y_pred - center_y_target)**2
        
        # 计算v项（长宽比一致性）
        v = (4 / (math.pi**2)) * torch.pow(
            torch.atan(w_target / h_target) - torch.atan(w_pred / h_pred), 2
        )
        
        # 计算alpha（平衡因子）（添加eps避免除零）
        alpha = v / ((1 - iou) + v + self.eps)
        
        # 计算DIoU（添加eps避免除零）
        diou = iou - (center_dist2 / (c2 + self.eps))
        
        # 计算CIoU
        ciou = diou - alpha * v
        
        # 裁剪值域，避免极端值
        ciou = torch.clamp(ciou, min=-1.0 + self.eps, max=1.0 - self.eps)
        
        # 转换为损失（1 - CIoU）
        ciou_loss = 1 - ciou
        
        return ciou_loss.mean()