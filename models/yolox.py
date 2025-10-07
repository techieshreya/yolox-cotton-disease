import torch
import torch.nn as nn
import math

# Basic building blocks -----------------------------------------------------------


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class ECA(nn.Module):
    """Efficient Channel Attention - lighter alternative to CBAM"""

    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CoordAttention(nn.Module):
    """Coordinate Attention for better localization"""

    def __init__(self, in_channels, reduction=16):
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = SiLU()

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_h * a_w


class DropBlock2D(nn.Module):
    """DropBlock for regularization - better than dropout for CNNs"""

    def __init__(self, drop_prob=0.1, block_size=7):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x

        # Sample mask
        gamma = self.drop_prob / (self.block_size**2)
        mask = torch.bernoulli(torch.ones_like(x[:, :1, :, :]) * gamma)

        # Expand mask to block size
        block_mask = torch.nn.functional.max_pool2d(
            mask, stride=1, kernel_size=self.block_size, padding=self.block_size // 2
        )

        # Ensure mask matches input size exactly
        if block_mask.shape[2] != x.shape[2] or block_mask.shape[3] != x.shape[3]:
            block_mask = torch.nn.functional.interpolate(
                block_mask, size=(x.shape[2], x.shape[3]), mode="nearest"
            )

        # Invert mask (1 = keep, 0 = drop)
        block_mask = 1 - block_mask

        # Normalize and apply
        normalize_factor = block_mask.numel() / (block_mask.sum() + 1e-7)
        return x * block_mask * normalize_factor


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> SiLU block"""

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
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
        x = torch.cat(
            (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1
        )
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

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
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
    """Enhanced CSP Bottleneck with dual attention and DropBlock"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        use_dropblock=True,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, 1)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, 1)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, 1)
        self.m = nn.Sequential(
            *[
                Bottleneck(hidden_channels, hidden_channels, shortcut, expansion=1.0)
                for _ in range(n)
            ]
        )

        # Dual attention: ECA for efficiency + Coordinate for localization
        self.eca = ECA(out_channels)
        self.coord_attn = CoordAttention(out_channels, reduction=16)

        # DropBlock for regularization
        self.dropblock = (
            DropBlock2D(drop_prob=0.1, block_size=7) if use_dropblock else nn.Identity()
        )

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        x = self.conv3(x)

        # Apply dual attention
        x = self.eca(x)
        x = self.coord_attn(x)
        x = self.dropblock(x)
        return x


class SimSPPF(nn.Module):
    """Enhanced SPPF with multiple kernel sizes for better multi-scale features"""

    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = BaseConv(in_channels, c_, 1, 1)

        # Multiple pooling sizes for better receptive fields
        self.m1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.m2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.m3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

        self.cv2 = BaseConv(c_ * 4, out_channels, 1, 1)
        self.attn = ECA(out_channels)

    def forward(self, x):
        x = self.cv1(x)
        with torch.cuda.amp.autocast(enabled=False):
            y1 = self.m1(x)
            y2 = self.m2(x)
            y3 = self.m3(x)
            out = self.cv2(torch.cat([x, y1, y2, y3], 1))
            return self.attn(out)


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
            CSPLayer(
                base_channels * 16, base_channels * 16, n=base_depth, shortcut=False
            ),
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
    """Enhanced PANet FPN with feature refinement"""

    def __init__(self, dep_mul, wid_mul):
        super().__init__()
        self.depth = dep_mul
        self.width = wid_mul
        in_channels = [
            int(self.width * 256),
            int(self.width * 512),
            int(self.width * 1024),
        ]

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.lateral_conv0 = BaseConv(in_channels[2], in_channels[1], 1, 1)
        self.C3_p4 = CSPLayer(
            in_channels[1] * 2,
            in_channels[1],
            n=max(round(self.depth * 3), 1),
            use_dropblock=True,
        )

        self.lateral_conv1 = BaseConv(in_channels[1], in_channels[0], 1, 1)
        self.C3_p3 = CSPLayer(
            in_channels[0] * 2,
            in_channels[0],
            n=max(round(self.depth * 3), 1),
            use_dropblock=True,
        )

        self.downsample_conv0 = BaseConv(in_channels[0], in_channels[0], 3, 2)
        self.C3_n3 = CSPLayer(
            in_channels[0] * 2,
            in_channels[1],
            n=max(round(self.depth * 3), 1),
            use_dropblock=True,
        )

        self.downsample_conv1 = BaseConv(in_channels[1], in_channels[1], 3, 2)
        self.C3_n4 = CSPLayer(
            in_channels[1] * 2,
            in_channels[2],
            n=max(round(self.depth * 3), 1),
            use_dropblock=True,
        )

        # Additional feature refinement for small objects (P3 level)
        self.refine_p3 = nn.Sequential(
            BaseConv(in_channels[0], in_channels[0], 3, 1), ECA(in_channels[0])
        )

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

        # Refine P3 for better small object detection
        pan_out2 = self.refine_p3(pan_out2)

        p_out1 = self.downsample_conv0(pan_out2)
        p_out1 = torch.cat([p_out1, fpn_out1], 1)
        pan_out1 = self.C3_n3(p_out1)

        p_out0 = self.downsample_conv1(pan_out1)
        p_out0 = torch.cat([p_out0, fpn_out0], 1)
        pan_out0 = self.C3_n4(p_out0)

        return (pan_out2, pan_out1, pan_out0)


class YOLOXHead(nn.Module):
    def __init__(
        self, num_classes, wid_mul=0.50, in_channels=[128, 256, 512], act="silu"
    ):
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
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        BaseConv(int(256 * wid_mul), int(256 * wid_mul), 3, 1),
                        BaseConv(int(256 * wid_mul), int(256 * wid_mul), 3, 1),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        BaseConv(int(256 * wid_mul), int(256 * wid_mul), 3, 1),
                        BaseConv(int(256 * wid_mul), int(256 * wid_mul), 3, 1),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    int(256 * wid_mul), self.n_anchors * self.num_classes, 1, 1, 0
                )
            )
            self.reg_preds.append(nn.Conv2d(int(256 * wid_mul), 4, 1, 1, 0))
            self.obj_preds.append(
                nn.Conv2d(int(256 * wid_mul), self.n_anchors * 1, 1, 1, 0)
            )

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
    def __init__(self, num_classes=2, phi="s"):
        super().__init__()
        depth_dict = {"s": 0.33, "m": 0.67, "l": 1.00, "x": 1.33}
        width_dict = {"s": 0.50, "m": 0.75, "l": 1.00, "x": 1.25}
        dep_mul, wid_mul = depth_dict[phi], width_dict[phi]

        self.backbone = CSPDarknet(dep_mul, wid_mul)
        self.neck = YOLOPAFPN(dep_mul, wid_mul)
        in_channels = [int(wid_mul * 256), int(wid_mul * 512), int(wid_mul * 1024)]
        self.head = YOLOXHead(num_classes, wid_mul, in_channels)
        self.stride = torch.tensor([8.0, 16.0, 32.0])  # Add stride for loss calculation

        # Initialize head biases for better initial convergence
        self.head.initialize_biases(prior_prob=0.01)

    def forward(self, x):
        fpn_features = self.backbone(x)
        pan_features = self.neck(fpn_features)
        outputs = self.head(pan_features)
        return outputs
