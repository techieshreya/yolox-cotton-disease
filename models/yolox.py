import torch
import torch.nn as nn

class YOLOX(nn.Module):
    def __init__(self, num_classes=2, depth_multiple=0.33, width_multiple=0.50):
        super().__init__()
        self.num_classes = num_classes
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        
        # NOTE: Based on your architecture, the final feature map is downsampled 4 times (2*2*2*2).
        # If the input is 416x416, the feature map is 26x26. Stride = 416/26 = 16.
        self.stride = 16.0

        # Calculate scaled channels
        base_channels = int(64 * width_multiple)
        neck_channels = int(256 * width_multiple)

        # Backbone (simplified CSPDarknet)
        self.backbone = nn.Sequential(
            # Stem
            nn.Conv2d(3, base_channels, 6, 2, 2, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),

            # Stage 1 (stride 2)
            self._make_layer(base_channels, base_channels*2, 3),

            # Stage 2 (stride 4 -> 8)
            self._make_layer(base_channels*2, base_channels*4, 3),

            # Stage 3 (stride 8 -> 16)
            self._make_layer(base_channels*4, base_channels*8, 3),
        )

        # Neck (FPN - simplified)
        self.neck = nn.Sequential(
            nn.Conv2d(base_channels*8, neck_channels, 1, bias=False),
            nn.BatchNorm2d(neck_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(neck_channels, neck_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(neck_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Head
        self.head = nn.ModuleDict({
            'cls': nn.Conv2d(neck_channels, num_classes, 1),
            'reg': nn.Conv2d(neck_channels, 4, 1),   # 4 for cx, cy, w, h
            'obj': nn.Conv2d(neck_channels, 1, 1)    # 1 for objectness
        })

        self._initialize_head()

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            )
        )
        for _ in range(max(round(blocks * self.depth_multiple), 1)):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True)
                )
            )
        return nn.Sequential(*layers)

    def _initialize_head(self):
        bias_value = -torch.log(torch.tensor((1 - 0.01) / 0.01))
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Set specific bias for the objectness layer
        nn.init.constant_(self.head['obj'].bias, bias_value)

    def decode_outputs(self, reg_pred_map):
        """
        Decodes the raw regression output map into absolute coordinates.
        This is the crucial step for evaluation and inference.
        """
        batch_size, _, h, w = reg_pred_map.shape
        
        # Create a grid of (x, y) coordinates for the feature map
        yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing="ij")
        # Stack to get (h, w, 2) and flatten to (h*w, 2)
        grid = torch.stack((xv, yv), 2).view(1, h*w, 2).to(reg_pred_map.device)
        
        # Reshape raw regression predictions from (B, 4, H, W) to (B, H*W, 4)
        reg_pred_flat = reg_pred_map.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        
        # Apply the decoding formula
        # reg_pred_flat[..., :2] contains (tx, ty)
        # reg_pred_flat[..., 2:] contains (tw, th)
        # cx, cy = (grid_coord + pred_offset) * stride
        decoded_boxes_center = (reg_pred_flat[..., :2] + grid) * self.stride
        
        # w, h = exp(pred_log_scale) * stride
        decoded_boxes_wh = torch.exp(reg_pred_flat[..., 2:]) * self.stride
        
        # Combine into [cx, cy, w, h] format
        decoded_boxes = torch.cat((decoded_boxes_center, decoded_boxes_wh), dim=-1)
        
        # Reshape back to feature map format (B, 4, H, W) to match the training script expectation
        return decoded_boxes.permute(0, 2, 1).reshape(batch_size, 4, h, w)

    def forward(self, x):
        features = self.backbone(x)
        features = self.neck(features)

        cls_pred = self.head['cls'](features)
        reg_pred = self.head['reg'](features)
        obj_pred = self.head['obj'](features)
        
        if self.training:
            # During training, return raw outputs for the loss function
            return cls_pred, reg_pred, obj_pred
        else:
            # During evaluation, return raw outputs AND decoded box predictions
            # The training script uses the decoded boxes for post-processing and mAP
            decoded_reg_pred = self.decode_outputs(reg_pred)
            return cls_pred, reg_pred, obj_pred, decoded_reg_pred