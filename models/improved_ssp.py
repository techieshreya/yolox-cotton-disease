import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ImprovedSPP(nn.Module):
    """
    Improved Spatial Pyramid Pooling (SPP) block with skip connections.
    Based on the paper: "Handling Severity Levels of Multiple Co-Occurring Cotton Plant Diseases Using Improved YOLOX Model"
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Store input channels for skip connection
        self.in_channels = in_channels
        
        # Initial 1x1 Conv to reduce channels
        self.conv1 = BaseConv(in_channels, in_channels // 2, 1, 1)
        
        # Multiple Max Pooling layers with different kernel sizes
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.maxpool7 = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.maxpool9 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        
        # Channel attention for feature refinement
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels // 2 * 5, in_channels // 2 * 5 // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2 * 5 // 16, in_channels // 2 * 5, 1),
            nn.Sigmoid()
        )
        
        # Final 1x1 Conv
        self.conv2 = BaseConv(in_channels // 2 * 5, out_channels, 1, 1)
        
        # Skip connection projection if input/output channels differ
        if in_channels != out_channels:
            self.skip_conv = BaseConv(in_channels, out_channels, 1, 1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        """
        Forward pass of the Improved SPP block with skip connections.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Output feature map.
        """
        # Store input for skip connection
        skip = x
        
        # Initial convolution
        x = self.conv1(x)
        
        # Multiple Max Pooling operations
        mp3 = self.maxpool3(x)
        mp5 = self.maxpool5(x)
        mp7 = self.maxpool7(x)
        mp9 = self.maxpool9(x)
        
        # Concatenate all features
        x = torch.cat([x, mp3, mp5, mp7, mp9], dim=1)
        
        # Apply channel attention
        attention = self.channel_attention(x)
        x = x * attention
        
        # Final convolution
        x = self.conv2(x)
        
        # Add skip connection
        skip = self.skip_conv(skip)
        x = x + skip
        
        return x


if __name__ == '__main__':
    # --- Example Usage ---
    in_channels = 256  # Example input channels
    out_channels = 1024 # Example output channels
    batch_size = 1
    input_size = 20  # Example feature map size (20x20)

    improved_spp = ImprovedSPP(in_channels, out_channels)
    input_tensor = torch.randn(batch_size, in_channels, input_size, input_size)

    output_tensor = improved_spp(input_tensor)

    print("Input Tensor Shape:", input_tensor.shape)
    print("Output Tensor Shape:", output_tensor.shape)