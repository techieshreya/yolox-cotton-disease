import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedSPP(nn.Module):
    """
    Improved Spatial Pyramid Pooling (SPP) block.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # --- Initial 1x1 Conv ---
        self.conv1 = Conv(in_channels, 512, kernel_size=1)

        # --- Max Pooling layers ---
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.maxpool7 = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.maxpool9 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)

        # --- Final 1x1 Conv ---
        self.conv2 = Conv(512 * 5, out_channels, kernel_size=1)  # 5 inputs from Conv1 and 4 MaxPools

    def forward(self, x):
        """
        Forward pass of the Improved SPP block.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Output feature map.
        """
        x = self.conv1(x)

        # --- Max Pooling ---
        mp3 = self.maxpool3(x)
        mp5 = self.maxpool5(x)
        mp7 = self.maxpool7(x)
        mp9 = self.maxpool9(x)

        # --- Concatenate ---
        x = torch.cat([x, mp3, mp5, mp7, mp9], dim=1)

        # --- Final Conv ---
        x = self.conv2(x)

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