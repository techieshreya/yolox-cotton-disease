import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================================
# ENHANCED BUILDING BLOCKS FOR YOLOX-M++
# ============================================================================

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

class EvoNorm(nn.Module):
    """EvoNorm for better normalization"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
    def forward(self, x):
        if self.training:
            var = torch.var(x, dim=(0, 2, 3), keepdim=True)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            var = self.running_var.view(1, -1, 1, 1)
        
        std = torch.sqrt(var + self.eps)
        x = x / std
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

class GroupNorm(nn.Module):
    """Group Normalization with SiLU activation"""
    def __init__(self, num_channels, num_groups=32):
        super().__init__()
        # Ensure num_channels is divisible by num_groups by reducing groups if needed
        groups = min(num_groups, num_channels)
        while groups > 1 and (num_channels % groups) != 0:
            groups -= 1
        self.gn = nn.GroupNorm(groups, num_channels)
        self.act = SiLU()
        
    def forward(self, x):
        return self.act(self.gn(x))

class GhostConv(nn.Module):
    """Ghost Convolution for efficiency"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / 2)
        # Ensure cheap operation produces the remaining channels (non-zero)
        new_channels = max(out_channels - init_channels, 1)
        
        self.primary_conv = nn.Conv2d(in_channels, init_channels, kernel_size, stride, padding, groups=1, bias=False)
        self.cheap_operation = nn.Conv2d(init_channels, new_channels, kernel_size=1, groups=init_channels, bias=False)
        
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = GroupNorm(out_channels)
        self.act = SiLU()
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class DropBlock2D(nn.Module):
    """DropBlock for regularization"""
    def __init__(self, drop_prob=0.1, block_size=7):
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x

        gamma = self.drop_prob / (self.block_size**2)
        mask = torch.bernoulli(torch.ones_like(x[:, :1, :, :]) * gamma)
        
        block_mask = torch.nn.functional.max_pool2d(
            mask, stride=1, kernel_size=self.block_size, padding=self.block_size // 2
        )
        
        if block_mask.shape[2] != x.shape[2] or block_mask.shape[3] != x.shape[3]:
            block_mask = torch.nn.functional.interpolate(
                block_mask, size=(x.shape[2], x.shape[3]), mode="nearest"
            )
        
        block_mask = 1 - block_mask
        normalize_factor = block_mask.numel() / (block_mask.sum() + 1e-7)
        return x * block_mask * normalize_factor

class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_planes, ratio=16):
        super().__init__()
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
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super().__init__()
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
    """Convolutional Block Attention Module"""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block for modern architecture"""
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropBlock2D(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x

class CSPXBlock(nn.Module):
    """CSP-X Block with ConvNeXt integration"""
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = GhostConv(in_channels, hidden_channels, 1, 1)
        self.conv2 = GhostConv(in_channels, hidden_channels, 1, 1)
        self.conv3 = GhostConv(2 * hidden_channels, out_channels, 1, 1)
        
        # ConvNeXt blocks instead of standard bottlenecks
        self.m = nn.Sequential(*[
            ConvNeXtBlock(hidden_channels, drop_path=0.1)
            for _ in range(n)
        ])
        
        # CBAM attention
        self.cbam = CBAM(out_channels)
        self.dropblock = DropBlock2D(drop_prob=0.1, block_size=7)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        x = self.conv3(x)
        x = self.cbam(x)
        x = self.dropblock(x)
        return x

class DualSPP(nn.Module):
    """Dual Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        c_ = in_channels // 2
        
        # First SPP branch
        self.cv1 = GhostConv(in_channels, c_, 1, 1)
        self.m1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.m2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.m3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.cv2 = GhostConv(c_ * 4, out_channels, 1, 1)
        
        # Second SPP branch with different kernel sizes
        self.cv3 = GhostConv(in_channels, c_, 1, 1)
        self.m4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.m5 = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.m6 = nn.MaxPool2d(kernel_size=11, stride=1, padding=5)
        self.cv4 = GhostConv(c_ * 4, out_channels, 1, 1)
        
        # Fusion
        self.fusion = GhostConv(out_channels * 2, out_channels, 1, 1)
        self.attention = CBAM(out_channels)

    def forward(self, x):
        # First branch
        x1 = self.cv1(x)
        y1 = self.m1(x1)
        y2 = self.m2(x1)
        y3 = self.m3(x1)
        out1 = self.cv2(torch.cat([x1, y1, y2, y3], 1))
        
        # Second branch
        x2 = self.cv3(x)
        y4 = self.m4(x2)
        y5 = self.m5(x2)
        y6 = self.m6(x2)
        out2 = self.cv4(torch.cat([x2, y4, y5, y6], 1))
        
        # Fusion
        out = torch.cat([out1, out2], 1)
        out = self.fusion(out)
        out = self.attention(out)
        return out

class BiFPN(nn.Module):
    """Bidirectional Feature Pyramid Network with learnable weights and channel alignment"""
    def __init__(self, channels_list, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.channels_list = channels_list
        # Unify all feature maps to the same channel dimension for add-based fusion
        self.out_c = channels_list[0]  # use smallest level channels as common dimension

        # Per-level 1x1 conv to align channels
        self.adjust_convs = nn.ModuleList([
            nn.Conv2d(c_in, self.out_c, kernel_size=1, stride=1, padding=0, bias=False)
            for c_in in channels_list
        ])

        # Learnable weights for feature fusion
        self.weights = nn.ParameterList([
            nn.Parameter(torch.ones(2, dtype=torch.float32)) for _ in range(num_layers)
        ])

        # Top-down and Bottom-up pathway convs (keep channel dim constant)
        self.top_down_convs = nn.ModuleList([
            DepthwiseSeparableConv(self.out_c, self.out_c) for _ in range(len(channels_list) - 1)
        ])
        self.bottom_up_convs = nn.ModuleList([
            DepthwiseSeparableConv(self.out_c, self.out_c) for _ in range(len(channels_list) - 1)
        ])

    def forward(self, features):
        # Align channels for all input features first
        feats = [adj(f) for adj, f in zip(self.adjust_convs, features)]

        # Top-down pathway
        top_down_features = []
        for i in range(len(feats)):
            if i == 0:
                top_down_features.append(feats[i])
            else:
                weight = F.relu(self.weights[i-1])
                weight = weight / (weight.sum() + 1e-8)
                fused = weight[0] * feats[i] + weight[1] * F.interpolate(
                    top_down_features[-1], size=feats[i].shape[2:], mode='nearest'
                )
                top_down_features.append(self.top_down_convs[i-1](fused))

        # Bottom-up pathway
        bottom_up_features = []
        for i in range(len(top_down_features) - 1, -1, -1):
            if i == len(top_down_features) - 1:
                bottom_up_features.append(top_down_features[i])
            else:
                weight = F.relu(self.weights[i])
                weight = weight / (weight.sum() + 1e-8)
                fused = weight[0] * top_down_features[i] + weight[1] * F.interpolate(
                    bottom_up_features[-1], size=top_down_features[i].shape[2:], mode='nearest'
                )
                bottom_up_features.append(self.bottom_up_convs[i](fused))

        return list(reversed(bottom_up_features))

class ASFF(nn.Module):
    """Adaptive Spatial Feature Fusion"""
    def __init__(self, level, channels):
        super().__init__()
        self.level = level
        self.channels = channels
        
        # Learnable weights for different levels
        self.weight_level_0 = GhostConv(channels, 1, 1)
        self.weight_level_1 = GhostConv(channels, 1, 1)
        self.weight_level_2 = GhostConv(channels, 1, 1)
        
        # Feature refinement
        self.refine = GhostConv(channels, channels, 3, 1, 1)

    def forward(self, x_level_0, x_level_1, x_level_2):
        # Resize features to same size
        size_level_0 = x_level_0.shape[2:]
        
        level_0_weight_v = self.weight_level_0(x_level_0)
        level_1_weight_v = self.weight_level_1(F.interpolate(x_level_1, size=size_level_0, mode='nearest'))
        level_2_weight_v = self.weight_level_2(F.interpolate(x_level_2, size=size_level_0, mode='nearest'))
        
        # Softmax normalization
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = F.softmax(levels_weight_v, dim=1)
        
        # Weighted fusion
        fused_out = x_level_0 * levels_weight[:, 0:1, :, :] + \
                   F.interpolate(x_level_1, size=size_level_0, mode='nearest') * levels_weight[:, 1:2, :, :] + \
                   F.interpolate(x_level_2, size=size_level_0, mode='nearest') * levels_weight[:, 2:, :, :]
        
        return self.refine(fused_out)

class DyHead(nn.Module):
    """Dynamic Head for better feature representation"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Scale attention
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply different types of attention
        spatial_att = self.spatial_attention(x)
        channel_att = self.channel_attention(x)
        scale_att = self.scale_attention(x)
        
        # Combine attentions
        x = x * spatial_att * channel_att * scale_att
        return x

class DualBranchHead(nn.Module):
    """Dual-Branch Head for Detection + Severity"""
    def __init__(self, num_classes, num_severity_levels=5, wid_mul=0.75, in_channels=[192, 384, 768, 1536]):
        super().__init__()
        self.n_anchors = 1
        self.num_classes = num_classes
        self.num_severity_levels = num_severity_levels
        
        # Detection branch
        self.detection_convs = nn.ModuleList()
        self.detection_preds = nn.ModuleList()
        # Regression branch
        self.reg_stems = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        
        # Severity branch
        self.severity_convs = nn.ModuleList()
        self.severity_preds = nn.ModuleList()
        
        # Shared objectness
        self.obj_convs = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        
        # Stems
        self.detection_stems = nn.ModuleList()
        self.severity_stems = nn.ModuleList()
        self.obj_stems = nn.ModuleList()
        
        for i, in_ch in enumerate(in_channels):
            # Detection branch
            self.detection_stems.append(GhostConv(in_ch, int(256 * wid_mul), 1, 1))
            self.detection_convs.append(nn.Sequential(
                GhostConv(int(256 * wid_mul), int(256 * wid_mul), 3, 1, 1),
                GhostConv(int(256 * wid_mul), int(256 * wid_mul), 3, 1, 1),
            ))
            self.detection_preds.append(
                nn.Conv2d(int(256 * wid_mul), self.n_anchors * self.num_classes, 1, 1, 0)
            )

            # Regression branch (4 bbox channels)
            self.reg_stems.append(GhostConv(in_ch, int(256 * wid_mul), 1, 1))
            self.reg_convs.append(nn.Sequential(
                GhostConv(int(256 * wid_mul), int(256 * wid_mul), 3, 1, 1),
                GhostConv(int(256 * wid_mul), int(256 * wid_mul), 3, 1, 1),
            ))
            self.reg_preds.append(
                nn.Conv2d(int(256 * wid_mul), self.n_anchors * 4, 1, 1, 0)
            )
            
            # Severity branch
            self.severity_stems.append(GhostConv(in_ch, int(256 * wid_mul), 1, 1))
            self.severity_convs.append(nn.Sequential(
                GhostConv(int(256 * wid_mul), int(256 * wid_mul), 3, 1, 1),
                GhostConv(int(256 * wid_mul), int(256 * wid_mul), 3, 1, 1),
            ))
            self.severity_preds.append(
                nn.Conv2d(int(256 * wid_mul), self.n_anchors * self.num_severity_levels, 1, 1, 0)
            )
            
            # Objectness branch
            self.obj_stems.append(GhostConv(in_ch, int(256 * wid_mul), 1, 1))
            self.obj_convs.append(nn.Sequential(
                GhostConv(int(256 * wid_mul), int(256 * wid_mul), 3, 1, 1),
                GhostConv(int(256 * wid_mul), int(256 * wid_mul), 3, 1, 1),
            ))
            self.obj_preds.append(
                nn.Conv2d(int(256 * wid_mul), self.n_anchors * 1, 1, 1, 0)
            )

    def forward(self, fpn_feats):
        detection_outputs = []
        reg_outputs = []
        severity_outputs = []
        obj_outputs = []
        
        for i, feat in enumerate(fpn_feats):
            # Detection branch
            det_feat = self.detection_stems[i](feat)
            det_feat = self.detection_convs[i](det_feat)
            det_output = self.detection_preds[i](det_feat)
            
            # Regression branch
            reg_feat = self.reg_stems[i](feat)
            reg_feat = self.reg_convs[i](reg_feat)
            reg_output = self.reg_preds[i](reg_feat)

            # Severity branch
            sev_feat = self.severity_stems[i](feat)
            sev_feat = self.severity_convs[i](sev_feat)
            sev_output = self.severity_preds[i](sev_feat)
            
            # Objectness branch
            obj_feat = self.obj_stems[i](feat)
            obj_feat = self.obj_convs[i](obj_feat)
            obj_output = self.obj_preds[i](obj_feat)
            
            detection_outputs.append(det_output)
            reg_outputs.append(reg_output)
            severity_outputs.append(sev_output)
            obj_outputs.append(obj_output)
        
        return detection_outputs, severity_outputs, obj_outputs, reg_outputs

# ============================================================================
# YOLOX-M++ ARCHITECTURE
# ============================================================================

class Focus(nn.Module):
    """Focus width and height information into channel space."""
    def __init__(self, in_channels, out_channels, ksize=1, stride=1):
        super().__init__()
        self.conv = GhostConv(in_channels * 4, out_channels, ksize, stride)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1
        )
        return self.conv(x)

class CSPDarknetX(nn.Module):
    """Enhanced CSPDarknet with CSP-X and ConvNeXt"""
    def __init__(self, dep_mul, wid_mul, out_features=("dark2", "dark3", "dark4", "dark5")):
        super().__init__()
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        self.stem = Focus(3, base_channels, ksize=3)
        self.dark2 = nn.Sequential(
            DepthwiseSeparableConv(base_channels, base_channels * 2, 3, 2),
            CSPXBlock(base_channels * 2, base_channels * 2, n=base_depth),
        )
        self.dark3 = nn.Sequential(
            DepthwiseSeparableConv(base_channels * 2, base_channels * 4, 3, 2),
            CSPXBlock(base_channels * 4, base_channels * 4, n=base_depth * 3),
        )
        self.dark4 = nn.Sequential(
            DepthwiseSeparableConv(base_channels * 4, base_channels * 8, 3, 2),
            CSPXBlock(base_channels * 8, base_channels * 8, n=base_depth * 3),
        )
        self.dark5 = nn.Sequential(
            DepthwiseSeparableConv(base_channels * 8, base_channels * 16, 3, 2),
            DualSPP(base_channels * 16, base_channels * 16),
            CSPXBlock(
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

class YOLOXMPPNeck(nn.Module):
    """Enhanced Neck with BiFPN + ASFF"""
    def __init__(self, dep_mul, wid_mul):
        super().__init__()
        self.depth = dep_mul
        self.width = wid_mul
        # Match backbone outputs: dark2,3,4,5 channels = base_channels*[2,4,8,16]
        # base_channels = int(wid_mul * 64) -> channels = wid_mul * [128, 256, 512, 1024]
        in_channels = [
            int(self.width * 128),   # P2 (dark2)
            int(self.width * 256),   # P3 (dark3)
            int(self.width * 512),   # P4 (dark4)
            int(self.width * 1024),  # P5 (dark5)
        ]

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Lateral and reduction convolutions for top-down pathway
        self.lateral_p5 = GhostConv(in_channels[3], in_channels[2], 1, 1)
        self.reduce_p4 = GhostConv(in_channels[2] * 2, in_channels[2], 1, 1)
        self.reduce_p3 = GhostConv(in_channels[2] + in_channels[1], in_channels[1], 1, 1)
        self.reduce_p2 = GhostConv(in_channels[1] + in_channels[0], in_channels[0], 1, 1)

        # BiFPN
        self.bifpn = BiFPN(in_channels, num_layers=3)
        
        # ASFF for feature fusion (use unified channel dim from BiFPN)
        self.asff_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.asff_convs.append(ASFF(i, self.bifpn.out_c))

    def forward(self, input):
        out_features = (input["dark2"], input["dark3"], input["dark4"], input["dark5"])
        p2, p3, p4, p5 = out_features

        # Top-down pathway: reduce to expected channels after each concat
        td_p4 = self.lateral_p5(p5)
        td_p4 = torch.cat([F.interpolate(td_p4, size=p4.shape[2:], mode='nearest'), p4], 1)
        td_p4 = self.reduce_p4(td_p4)

        td_p3 = torch.cat([F.interpolate(td_p4, size=p3.shape[2:], mode='nearest'), p3], 1)
        td_p3 = self.reduce_p3(td_p3)

        td_p2 = torch.cat([F.interpolate(td_p3, size=p2.shape[2:], mode='nearest'), p2], 1)
        td_p2 = self.reduce_p2(td_p2)

        # BiFPN processing
        features = [td_p2, td_p3, td_p4, p5]
        bifpn_features = self.bifpn(features)

        # ASFF fusion
        fused_features = []
        for i, feat in enumerate(bifpn_features):
            if i < len(self.asff_convs):
                # Use ASFF for multi-scale fusion
                if i == 0:
                    fused = self.asff_convs[i](feat, feat, feat)
                elif i == 1:
                    fused = self.asff_convs[i](feat, feat, bifpn_features[i-1])
                else:
                    fused = self.asff_convs[i](feat, bifpn_features[i-1], bifpn_features[i-2])
                fused_features.append(fused)
            else:
                fused_features.append(feat)

        return tuple(fused_features)

class YOLOXMPP(nn.Module):
    """YOLOX-M++ Model"""
    def __init__(self, num_classes=5, phi="m"):
        super().__init__()
        depth_dict = {"s": 0.33, "m": 0.67, "l": 1.00, "x": 1.33}
        width_dict = {"s": 0.50, "m": 0.75, "l": 1.00, "x": 1.25}
        dep_mul, wid_mul = depth_dict[phi], width_dict[phi]

        self.backbone = CSPDarknetX(dep_mul, wid_mul)
        self.neck = YOLOXMPPNeck(dep_mul, wid_mul)

# BiFPN unifies channels to neck.bifpn.out_c; the neck returns features with that channel count
        bifpn_out_c = self.neck.bifpn.out_c
# head expects one in_channel value per FPN level — use the unified channel for all scales
        head_in_channels = [bifpn_out_c] * 4

        self.head = DualBranchHead(num_classes, num_severity_levels=5, wid_mul=wid_mul, in_channels=head_in_channels)

        self.stride = torch.tensor([4.0, 8.0, 16.0, 32.0])  # 4 scales including P2

    def forward(self, x):
        fpn_features = self.backbone(x)
        pan_features = self.neck(fpn_features)
        detection_outputs, severity_outputs, obj_outputs, reg_outputs = self.head(pan_features)
        
        # Combine outputs for compatibility
        combined_outputs = []
        for i in range(len(detection_outputs)):
            # Standard YOLOX layout: [reg(4), obj(1), cls(num_classes)]
            combined = torch.cat([
                reg_outputs[i],
                obj_outputs[i],
                detection_outputs[i]
            ], dim=1)
            combined_outputs.append(combined)
        
        return combined_outputs
