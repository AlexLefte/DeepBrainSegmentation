import torch
from torch import nn
from src.models.submodules import *
from src.models.crfasrnn_pytorch.crfasrnn.crfrnn import CrfRnn


# Define the FCNN architecture
class FCnnModel(nn.Module):
    """
    Model architecture that replicates the FastSurfer F-CNN model.
    This model is based on "Competitive Dense Blocks"

    Attributes
    ----------
    Encoding blocks: enc1-4
    Bottleneck block: bottleneck
    Decoding blocks: dec4-1
    """

    def __init__(self,
                 params: dict):
        """
        Constructor
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

        # 2. Bottleneck
        self.bottleneck = CompetitiveDenseBlock(params=params, is_input=False, verbose=False)

        # 3. Defining the decoding sequence:
        self.dec4 = DecodingCDB(params=params)
        self.dec3 = DecodingCDB(params=params)
        self.dec2 = DecodingCDB(params=params)
        self.dec1 = DecodingCDB(params=params)

        # 4. Classifier
        self.classifier = ClassifierBlock(params=params)

        # Initialize the layers:
        self._initialize_weights()

        params["in_channels"] = in_channels

    def _initialize_weights(self):
        """
        Initializes the weights and biases of the U-Net constituent blocks
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Xavier uniform: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_
                # nn.init.xavier_uniform_(module.weight)
                # if module.bias is not None:
                #     nn.init.constant_(module.bias, 0.00001)
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)  # Initialize scale to 1
                nn.init.constant_(module.bias, 0)  # Initialize shift to 0

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # 1. Encoding
        x1, skip1, ind1 = self.enc1(x)
        x2, skip2, ind2 = self.enc2(x1)
        x3, skip3, ind3 = self.enc3(x2)
        x4, skip4, ind4 = self.enc4(x3)

        # 2. Bottleneck
        bottleneck_output = self.bottleneck(x4)

        # 3. Decoding (skip connections with maxout)
        x_dec4 = self.dec4(bottleneck_output, skip4, ind4)
        x_dec3 = self.dec3(x_dec4, skip3, ind3)
        x_dec2 = self.dec2(x_dec3, skip2, ind2)
        x_dec1 = self.dec1(x_dec2, skip1, ind1)

        # 4. Final convolution through the classifier
        x_final = self.classifier(x_dec1)
        return x_final


# Define the FCNN + CRF-RNN architectur
class FCnnCRF(FCnnModel):
    """
    Model architecture that consists of a U-net architecture (FCNN) + an RNN network (CRF)
    """
    def __init__(self,
                 params: dict,
                 image_dims: tuple):
        super().__init__(params)
        self.crf_rnn = CrfRnn(num_labels=params['num_classes'],
                              num_iterations=5)

    def forward(self,
                x):
        output = super(FCnnModel, self).forward(x)
        output = self.crf_rnn(x, output)
        return output

