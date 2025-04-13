import torch
import torch.nn as nn
from torchvision.models.resnet import resnet152
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from anchor_utils import generate_anchors, decode_yolo


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
    
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x
    
class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, stride, g=2):
        super(Conv, self).__init__()
        self.gc = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=g, bias=False)
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, input_data):
        return self.act(self.bn(self.gc(input_data) + self.pwc(input_data)))
    
    
class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c_, c2, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BBoxSupervisionDigitsResnet(nn.Module):
    
    def __init__(self, class_num=11):
        super(BBoxSupervisionDigitsResnet, self).__init__()
        net = nn.Sequential(*list(resnet152(pretrained=True).children())[:-2])
        self.backbone = net
        
        self.attn = CBAM(in_planes=2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # self.bbox_predictor1 = nn.Linear(1024, 4)  
        # self.bbox_predictor2 = nn.Linear(1024, 4)
        # self.bbox_predictor3 = nn.Linear(1024, 4)
        # self.bbox_predictor4 = nn.Linear(1024, 4)
        
        self.fc1 = nn.Linear(2048, class_num)
        self.fc2 = nn.Linear(2048, class_num)
        self.fc3 = nn.Linear(2048, class_num)
        self.fc4 = nn.Linear(2048, class_num)

    def forward(self, img):
        f4 = self.backbone(img)
        f4 = self.attn(f4)
        f4 = self.avgpool(f4).flatten(start_dim=1)
        
        # 字符分类
        c1 = self.fc1(f4)
        c2 = self.fc2(f4)
        c3 = self.fc3(f4)
        c4 = self.fc4(f4)

        return (c1, c2, c3, c4), None
    
    
class DigitsRNFPN(nn.Module):
    
    def __init__(self, class_num=11):
        super(DigitsRNFPN, self).__init__()
        
        self.backbone = resnet_fpn_backbone('resnet152', pretrained=True)
        
        self.attn = CBAM(in_planes=256)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.bp1 = nn.Linear(256, 4)
        self.bp2 = nn.Linear(256, 4)
        self.bp3 = nn.Linear(256, 4)
        self.bp4 = nn.Linear(256, 4)
        
        self.fc1 = nn.Linear(256, class_num)
        self.fc2 = nn.Linear(256, class_num)
        self.fc3 = nn.Linear(256, class_num)
        self.fc4 = nn.Linear(256, class_num)
        
    def forward(self, img):
        
        feat = self.backbone(img)
        f0, f1 = feat['0'], feat['1']
        f0 = self.pool(f0).flatten(start_dim=1)
        f1 = self.pool(self.attn(f1)).flatten(start_dim=1)
        
        b1 = self.bp1(f0)
        b2 = self.bp2(f0)
        b3 = self.bp3(f0)
        b4 = self.bp4(f0)
        
        c1 = self.fc1(f1)
        c2 = self.fc2(f1)
        c3 = self.fc3(f1)
        c4 = self.fc4(f1)
        
        return (c1, c2, c3, c4), (b1, b2, b3, b4)
    
class YOLODigitsRNFPN(nn.Module):
    """
    使用YOLO风格的anchor-based结构的数字识别模型
    """
    def __init__(self, class_num=11, num_anchors=3):
        super(YOLODigitsRNFPN, self).__init__()
        
        self.class_num = class_num
        self.num_anchors = num_anchors
        self.backbone = resnet_fpn_backbone('resnet152', pretrained=True)
        
        self.attn = CBAM(in_planes=256)
        
        # 定义固定的anchors大小 (宽、高)，可以根据数据集调整
        self.anchors = generate_anchors([
            [20, 30], [25, 35], [30, 40]  # 示例anchors, 应根据实际数据统计调整
        ])
        
        # YOLO风格的检测头
        # 每个anchor输出: tx, ty, tw, th, objectness, 类别概率
        self.detect_head = nn.Conv2d(256, num_anchors * (5 + class_num), 1)
        
        # 用于字符分类的全连接层
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, class_num)
        self.fc2 = nn.Linear(256, class_num)
        self.fc3 = nn.Linear(256, class_num)
        self.fc4 = nn.Linear(256, class_num)
        
    def forward(self, img):
        batch_size = img.shape[0]
        feat = self.backbone(img)
        f0, f1 = feat['0'], feat['1']  # 获取特征图
        
        # 应用CBAM注意力
        f1_attn = self.attn(f1)
        
        # 用于分类的特征
        f1_pool = self.pool(f1_attn).flatten(start_dim=1)
        
        # 分类预测
        c1 = self.fc1(f1_pool)
        c2 = self.fc2(f1_pool)
        c3 = self.fc3(f1_pool)
        c4 = self.fc4(f1_pool)
        
        # 检测预测
        # f0的特征图尺寸将决定grid的大小
        h, w = f0.shape[2], f0.shape[3]
        
        # 使用检测头进行预测
        detection = self.detect_head(f0)  # shape: (batch_size, num_anchors*(5+class_num), h, w)
        
        # 计算stride (原图到特征图的缩放比例)
        # 假设输入图像尺寸为 (H, W)，特征图尺寸为 (h, w)
        # stride就是 H/h 或 W/w
        stride = img.shape[2] / h  # 或img.shape[3] / w
        
        # 解码YOLO预测
        boxes, objectness, class_scores = decode_yolo(
            detection, 
            self.anchors, 
            stride, 
            (h, w)
        )
        
        # 根据objectness和类别分数选择最佳的4个框
        # 每个框代表一个数字位置
        pred_boxes = self._select_best_boxes(boxes, objectness, class_scores, 4)
        
        return (c1, c2, c3, c4), pred_boxes
    
    def _select_best_boxes(self, boxes, objectness, class_scores, num_boxes=4):
        """
        从所有预测框中选择最佳的num_boxes个框
        
        参数:
        - boxes: 预测框, shape (batch_size, num_anchors, h, w, 4)
        - objectness: 置信度, shape (batch_size, num_anchors, h, w)
        - class_scores: 类别分数, shape (batch_size, num_anchors, h, w, num_classes)
        - num_boxes: 选择的框数量，默认4
        
        返回:
        - 元组，包含num_boxes个最佳框(b1, b2, b3, b4)
        """
        batch_size = boxes.shape[0]
        device = boxes.device
        
        # 计算总得分(objectness * 最大类别分数)
        scores = objectness * class_scores.max(dim=-1)[0]  # (batch_size, num_anchors, h, w)
        
        # 将框和得分重塑为方便处理的形式
        boxes_reshaped = boxes.view(batch_size, -1, 4)  # (batch_size, num_anchors*h*w, 4)
        scores_reshaped = scores.view(batch_size, -1)  # (batch_size, num_anchors*h*w)
        
        # 创建结果容器
        result_boxes = []
        
        # 对每个样本，选择得分最高的num_boxes个框
        for i in range(batch_size):
            sample_scores = scores_reshaped[i]
            sample_boxes = boxes_reshaped[i]
            
            # 获取得分最高的num_boxes个框的索引
            if sample_scores.max() > 0:  # 确保有有效预测
                _, top_indices = torch.topk(sample_scores, min(num_boxes, len(sample_scores)))
                top_boxes = sample_boxes[top_indices]
            else:
                # 如果没有有效预测，返回零张量
                top_boxes = torch.zeros(num_boxes, 4, device=device)
            
            # 如果预测的框少于num_boxes，用零填充
            if len(top_boxes) < num_boxes:
                padding = torch.zeros(num_boxes - len(top_boxes), 4, device=device)
                top_boxes = torch.cat([top_boxes, padding], dim=0)
                
            # 分割成单独的框
            b1, b2, b3, b4 = top_boxes[:4]  # 取前4个框
            result_boxes.append((b1, b2, b3, b4))
            
        # 转置结果，使其与原来的模型输出格式一致: (b1, b2, b3, b4) 其中每个bi是(batch_size, 4)
        b1 = torch.stack([boxes[0] for boxes in result_boxes])
        b2 = torch.stack([boxes[1] for boxes in result_boxes])
        b3 = torch.stack([boxes[2] for boxes in result_boxes])
        b4 = torch.stack([boxes[3] for boxes in result_boxes])
        
        return (b1, b2, b3, b4)