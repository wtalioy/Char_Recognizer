import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from collections import OrderedDict

class DigitsResnet50(nn.Module):
    """
    ResNet50-based model for multi-digit recognition
    """
    def __init__(self, class_num=11):
        super(DigitsResnet50, self).__init__()
        self.net = resnet50(pretrained=True)
        self.net = nn.Sequential(*list(self.net.children())[:-1]) 
        self.cnn = self.net
        self.fc1 = nn.Linear(2048, class_num)
        self.fc2 = nn.Linear(2048, class_num)
        self.fc3 = nn.Linear(2048, class_num)
        self.fc4 = nn.Linear(2048, class_num)

    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        return c1, c2, c3, c4
    

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    

class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_linear = nn.Linear(hidden_size, hidden_size * 4)
        self.input_linear = nn.Linear(input_size, hidden_size * 4, bias=False)
        self.layer_norm = nn.ModuleList([LayerNorm(hidden_size) for _ in range(4)])
        self.layer_norm_c = nn.Identity()

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        gates = self.input_linear(x) + self.hidden_linear(h)
        i, f, g, o = gates.chunk(4, dim=-1)
        i, f, g, o = [layer_norm(layer) for layer, layer_norm in zip([i, f, g, o], self.layer_norm)]
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(self.layer_norm_c(c))
        return h, c


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, lstm_cell: nn.Module = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.lstm = lstm_cell

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor, prompt_num: int):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        if self.lstm is None:
            return x, None, None
        else:
            prompt = x[-prompt_num:, :, :]  # [prompt_num, batch_size, d_model]
            prompt = prompt.reshape(-1, prompt.shape[-1])  # [prompt_num * batch_size, d_model]
            x_origin = x[:-prompt_num, :, :]

            h, c = self.lstm(prompt, h, c)
            prompt = h.reshape(prompt_num, -1, h.shape[-1])  # [prompt_num, batch_size, d_model]    

            x = torch.cat([x_origin, prompt], dim=0)
            return x, h, c


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, lstm: bool = True):
        super().__init__()
        self.width = width
        self.layers = layers
        self.lstm_cell = LSTMCell(width, width) if lstm else None
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask, self.lstm_cell) for _ in range(layers)])

    def forward(self, x: torch.Tensor, prompt_num: int = 0):
        batch_size = x.shape[1]
        h = torch.zeros(prompt_num * batch_size, self.width, dtype=x.dtype, device=x.device)
        c = torch.zeros(prompt_num * batch_size, self.width, dtype=x.dtype, device=x.device)
        
        for block in self.resblocks:
            x, h, c = block(x, h, c, prompt_num)
            
        return x


class DigitsViT(nn.Module):
    def __init__(self, input_resolution=224, patch_size=14, width=1024, layers=24, heads=16, output_dim=768, prompt_num=3, class_num=11):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.class_embedding.require_grad = False
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.positional_embedding.require_grad = False
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, lstm=True)

        self.ln_post = LayerNorm(width)
        
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.proj.require_grad = False

        self.prompt_num = prompt_num
        self.prompt = nn.Parameter(torch.randn(self.prompt_num, width))

        self.fc1 = nn.Linear(width, class_num)
        self.fc2 = nn.Linear(width, class_num)
        self.fc3 = nn.Linear(width, class_num)
        self.fc4 = nn.Linear(width, class_num)

    def forward(self, x: torch.Tensor, micro_epoch: int = None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        if self.prompt is not None:
            micro_epoch = self.prompt_num if micro_epoch is None else micro_epoch
            prompt = self.prompt[:micro_epoch]
            prompt = prompt.expand(x.shape[0], -1, -1)
            prompt = prompt.to(x.device)
            x = torch.cat([x, prompt], dim=1)

            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x, prompt_num=micro_epoch)
            x = x.permute(1, 0, 2)  # LND -> NLD
        else:
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)

        x = self.ln_post(x[:, 0, :])
        # x = x @ self.proj

        c1 = self.fc1(x)
        c2 = self.fc2(x)
        c3 = self.fc3(x)
        c4 = self.fc4(x)

        return c1, c2, c3, c4


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
        log_soft = F.log_softmax(preds, dim=1)
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