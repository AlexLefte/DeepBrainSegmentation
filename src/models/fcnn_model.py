import torch
from torch import nn
from src.models.submodules import *


# Define the FCNN architecture
class FCnnModel(nn.Module):
    """
    Model architecture based on the FastSurfer model.
    Reference: https://github.com/deep-mi/FastSurfer.
    """

    def __init__(self,
                 params: dict):
        """
        Initializes the FCnn model with the given parameters.

        Parameters
        ----------
        params : dict
            Dictionary containing the configuration parameters for the block.
        """
        super().__init__()

        self.device = params["device"]
        filters = params["filters"]

        # 1. Defining the encoding sequence:
        self.enc1 = EncodingCDB(params=params, is_input=True)
        # From now on the input shape must be equal to the number of filters:
        in_channels = params["in_channels"]
        params["in_channels"] = filters
        self.enc2 = EncodingCDB(params=params)
        self.enc3 = EncodingCDB(params=params)
        self.enc4 = EncodingCDB(params=params)

        # 2. Bottleneck layer
        self.bottleneck = CompetitiveDenseBlock(params=params, is_input=False, verbose=False)

        # 3. Defining the decoding sequence:
        self.dec4 = DecodingCDB(params=params)
        self.dec3 = DecodingCDB(params=params)
        self.dec2 = DecodingCDB(params=params)
        self.dec1 = DecodingCDB(params=params)

        # 4. Classifier
        self.classifier = ClassifierBlock(params=params)

        # 5. Initialize the layer parameters
        self._initialize_weights()

        # Change input channels number back to the initial value
        params["in_channels"] = in_channels

    def _initialize_weights(self):
        """
        Initializes the weights and biases
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Initialize weights for convolutional layers using Kaiming normal distribution,
                # suitable for layers with ReLU or leaky ReLU activations.
                # Reference: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_
                # nn.init.xavier_uniform_(module.weight)
                # if module.bias is not None:
                #     nn.init.constant_(module.bias, 0.00001)
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                # Initialize weights for batch normalization layers to 1.
                nn.init.constant_(module.weight, 1)
                # Initialize biases for batch normalization layers to 0.
                nn.init.constant_(module.bias, 0)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # 1. Encoding path: progressively processes the input through encoding layers.
        x1, skip1, ind1 = self.enc1(x)
        x2, skip2, ind2 = self.enc2(x1)
        x3, skip3, ind3 = self.enc3(x2)
        x4, skip4, ind4 = self.enc4(x3)

        # 2. Bottleneck: processes the deepest features, acting as a bridge between encoding and decoding.
        bottleneck_output = self.bottleneck(x4)

        # 3. Decoding path: reconstructs the features back to the original resolution, utilizing skip connections.
        x_dec4 = self.dec4(bottleneck_output, skip4, ind4)
        x_dec3 = self.dec3(x_dec4, skip3, ind3)
        x_dec2 = self.dec2(x_dec3, skip2, ind2)
        x_dec1 = self.dec1(x_dec2, skip1, ind1)

        # 4. Final classification layer: applies the final convolution to produce the output.
        x_final = self.classifier(x_dec1)
        return x_final