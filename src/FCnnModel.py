import torch
from torch import nn


# Define the first Competitive Dense Block (CDB)
class CompetitiveDenseBlock(nn.Module):
    """
    Each block is composed of three sequences of parametric rectified linear unit (PReLU), convolution (Conv) and
    normalization (BN) except for the very first encoder block. In the first block, the PReLU is replaced
    with a BN to normalize the raw inputs.
    """
    def __init__(self, in_channels, out_channels, is_first=False):
        super().__init__()
        self.is_first = is_first
        self.seq1 = nn.Sequential(
            nn.BatchNorm2d(out_channels) if is_first else nn.PReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.seq2 = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.seq3 = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = torch.max(x, self.seq1(x)) if self.is_first else self.seq1(x)
        out = torch.max(out, self.seq2(out))
        out = self.seq3(out)
        return out


# Define the F-CNN architecture
class FCnnModel(nn.Module):
    """
    Model architecture that replicates the FastSurfer F-CNN model.
    This model is based on "competitive dense blocks"
    """

    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()

        self.enc1 = CompetitiveDenseBlock(input_shape, output_shape)
        self.cdb1 = CompetitiveDenseBlock(64, 128)
        self.cdb2 = CompetitiveDenseBlock(128, 256)
        self.cdb3 = CompetitiveDenseBlock(256, 512)
        self.cdb4 = CompetitiveDenseBlock(512, 1024)
        self.bottleneck = nn.Conv2d(1024, 1024, kernel_size=1)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # Decoder Competitive Dense Blocks
        self.cdb_dec4 = CompetitiveDenseBlock(512, 256)
        self.cdb_dec3 = CompetitiveDenseBlock(256, 128)
        self.cdb_dec2 = CompetitiveDenseBlock(128, 64)
        self.cdb_dec1 = CompetitiveDenseBlock(64, 32)

        self.final_conv = nn.Conv2d(64, output_shape, kernel_size=1)

    def forward(self, x):
        # Encoding
        x1 = self.enc1(x)
        x2 = self.cdb1(x1)
        x3 = self.cdb2(x2)
        x4 = self.cdb3(x3)
        x5 = self.cdb4(x4)
        x5 = self.bottleneck(x5)

        # Decoder (skip connections with maxout)
        x_up4 = self.upconv4(x5)
        x_max4 = torch.max(x_up4, x4)
        x_dec4 = self.cbd_dec4(x_max4)
        x_up3 = self.upconv3(x_dec4)
        x_max3 = torch.max(x_up3, x3)
        x_dec3 = self.cbd_dec3(x_max3)
        x_up2 = self.upconv2(x_dec3)
        x_max2 = torch.max(x_up2, x2)
        x_dec2 = self.cbd_dec2(x_max2)
        x_up1 = self.upconv1(x_dec2)
        x_max1 = torch.max(x_up1, x1)
        x_dec1 = self.cbd_dec1(x_max1)

        # Final Convolution
        x_final = torch.softmax(self.final_conv(x_dec1), dim=1)
        return x_final
