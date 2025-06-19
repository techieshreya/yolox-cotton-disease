import torch
import torch.nn as nn
import math

# Basic building blocks -----------------------------------------------------------

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> SiLU block"""
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Focus(nn.Module):
    """Focus width and height information into channel space."""
    def __init__(self, in_channels, out_channels, ksize=1, stride=1):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1)
        return self.conv(x)

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, 1)
        self.conv2 = BaseConv(hidden_channels, out_channels, 3, 1)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
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
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, 1)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, 1)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(hidden_channels, hidden_channels, shortcut, expansion=1.0) for _ in range(n)])
        self.attn = CBAM(out_channels)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        x = self.conv3(x)
        return self.attn(x)

class SimSPPF(nn.Module):
    """Simplified SPPF layer for YOLOv6"""
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = BaseConv(in_channels, c_, 1, 1)
        self.cv2 = BaseConv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with torch.cuda.amp.autocast(enabled=False):
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

# YOLOX Architecture -------------------------------------------------------------

class CSPDarknet(nn.Module):
    def __init__(self, dep_mul, wid_mul, out_features=("dark3", "dark4", "dark5")):
        super().__init__()
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        self.stem = Focus(3, base_channels, ksize=3)
        self.dark2 = nn.Sequential(
            BaseConv(base_channels, base_channels * 2, 3, 2),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth),
        )
        self.dark3 = nn.Sequential(
            BaseConv(base_channels * 2, base_channels * 4, 3, 2),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3),
        )
        self.dark4 = nn.Sequential(
            BaseConv(base_channels * 4, base_channels * 8, 3, 2),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3),
        )
        self.dark5 = nn.Sequential(
            BaseConv(base_channels * 8, base_channels * 16, 3, 2),
            SimSPPF(base_channels * 16, base_channels * 16),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False),
        )
        self.out_features = out_features

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class YOLOPAFPN(nn.Module):
    def __init__(self, dep_mul, wid_mul):
        super().__init__()
        self.depth = dep_mul
        self.width = wid_mul
        in_channels = [int(self.width*256), int(self.width*512), int(self.width*1024)]

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        
        self.lateral_conv0 = BaseConv(in_channels[2], in_channels[1], 1, 1)
        self.C3_p4 = CSPLayer(in_channels[1] * 2, in_channels[1], n=max(round(self.depth*3), 1))

        self.lateral_conv1 = BaseConv(in_channels[1], in_channels[0], 1, 1)
        self.C3_p3 = CSPLayer(in_channels[0] * 2, in_channels[0], n=max(round(self.depth*3), 1))

        self.downsample_conv0 = BaseConv(in_channels[0], in_channels[0], 3, 2)
        self.C3_n3 = CSPLayer(in_channels[0] * 2, in_channels[1], n=max(round(self.depth*3), 1))
        
        self.downsample_conv1 = BaseConv(in_channels[1], in_channels[1], 3, 2)
        self.C3_n4 = CSPLayer(in_channels[1] * 2, in_channels[2], n=max(round(self.depth*3), 1))

    def forward(self, input):
        out_features = (input["dark3"], input["dark4"], input["dark5"])
        x2, x1, x0 = out_features

        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.upsample(fpn_out0)
        f_out0 = torch.cat([f_out0, x1], 1)
        f_out0 = self.C3_p4(f_out0)

        fpn_out1 = self.lateral_conv1(f_out0)
        f_out1 = self.upsample(fpn_out1)
        f_out1 = torch.cat([f_out1, x2], 1)
        pan_out2 = self.C3_p3(f_out1)

        p_out1 = self.downsample_conv0(pan_out2)
        p_out1 = torch.cat([p_out1, fpn_out1], 1)
        pan_out1 = self.C3_n3(p_out1)

        p_out0 = self.downsample_conv1(pan_out1)
        p_out0 = torch.cat([p_out0, fpn_out0], 1)
        pan_out0 = self.C3_n4(p_out0)

        return (pan_out2, pan_out1, pan_out0)


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, wid_mul=0.50, in_channels=[128, 256, 512], act="silu"):
        super().__init__()
        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        
        stems = [BaseConv(in_ch, int(256 * wid_mul), 1, 1) for in_ch in in_channels]

        for i in range(len(in_channels)):
            self.cls_convs.append(nn.Sequential(*[
                BaseConv(int(256 * wid_mul), int(256 * wid_mul), 3, 1),
                BaseConv(int(256 * wid_mul), int(256 * wid_mul), 3, 1),
            ]))
            self.reg_convs.append(nn.Sequential(*[
                BaseConv(int(256 * wid_mul), int(256 * wid_mul), 3, 1),
                BaseConv(int(256 * wid_mul), int(256 * wid_mul), 3, 1),
            ]))
            self.cls_preds.append(nn.Conv2d(int(256 * wid_mul), self.n_anchors * self.num_classes, 1, 1, 0))
            self.reg_preds.append(nn.Conv2d(int(256 * wid_mul), 4, 1, 1, 0))
            self.obj_preds.append(nn.Conv2d(int(256 * wid_mul), self.n_anchors * 1, 1, 1, 0))

        self.stems = nn.ModuleList(stems)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, fpn_feats):
        outputs = []
        for i, (feat, stem) in enumerate(zip(fpn_feats, self.stems)):
            with torch.cuda.amp.autocast(enabled=False):
                cls_x = stem(feat.float())
                reg_x = stem(feat.float())

                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)

                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)
                obj_output = self.obj_preds[i](reg_feat)
            
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)

        return outputs


class YOLOX(nn.Module):
    def __init__(self, num_classes=2, phi='s'):
        super().__init__()
        depth_dict = {'s': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33}
        width_dict = {'s': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25}
        dep_mul, wid_mul = depth_dict[phi], width_dict[phi]
        
        self.backbone = CSPDarknet(dep_mul, wid_mul)
        self.neck = YOLOPAFPN(dep_mul, wid_mul)
        in_channels = [int(wid_mul*256), int(wid_mul*512), int(wid_mul*1024)]
        self.head = YOLOXHead(num_classes, wid_mul, in_channels)
        self.stride = torch.tensor([8.0, 16.0, 32.0]) # Add stride for loss calculation

    def forward(self, x):
        fpn_features = self.backbone(x)
        pan_features = self.neck(fpn_features)
        outputs = self.head(pan_features)
        return outputs