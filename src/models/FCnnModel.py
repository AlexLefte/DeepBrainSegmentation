import torch
from torch import nn


# Define the first Competitive Dense Block (CDB)
class CompetitiveDenseBlock(nn.Module):
    """
    Each block is composed of three sequences of parametric rectified linear unit (PReLU), convolution (Conv) and
    normalization (BN) except for the very first encoder block. In the first block, the PReLU is replaced
    with a BN to normalize the raw inputs.

    Attributes
    ---------
    in_channels,
    out_channels,
    kernel_h & kernel_w,
    filters,
    padding_h & padding_w,
    stride;

    Methods
    -------
    forward
    """
    def __init__(self, params: dict, is_input=False, verbose=False):
        super().__init__()

        self.is_first = is_input
        in_channels = params["in_channels"]
        kernel_h = params["kernel_h"]
        kernel_w = params["kernel_w"]
        filters = params["filters"]
        stride = params["stride"]
        self.verbose = verbose

        # It is important to use padding in order to
        # ensure the output tensor has the same dimensions
        padding_h = (kernel_h - 1) // 2
        padding_w = (kernel_w - 1) // 2

        # Defining three distinct structures
        self.seq1 = nn.Sequential(
            nn.BatchNorm2d(in_channels) if is_input else nn.PReLU(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=filters,
                      kernel_size=(kernel_h, kernel_w),
                      stride=stride,
                      padding=(padding_h, padding_w)),
            nn.BatchNorm2d(filters)
        )

        self.seq2 = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_channels=filters,
                      out_channels=filters,
                      kernel_size=(kernel_h, kernel_w),
                      stride=stride,
                      padding=(padding_h, padding_w)),
            nn.BatchNorm2d(filters)
        )

        self.seq3 = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_channels=filters,
                      out_channels=filters,
                      kernel_size=(kernel_h, kernel_w),
                      stride=stride,
                      padding=(padding_h, padding_w)),
            nn.BatchNorm2d(filters)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input CDB:
                                            | - - - - - - - - - - - - - - - |
                                           |                               v
        Input -> (BN -> Conv(5x5) -> BN)  -> (PReLU -> Conv (5x5) -> BN) -> Maxout -> (PReLU -> Conv (1x1) -> BN)

        ----------------------------------------------------------------------------------------------------------

        Regular CDB:
                                               | - - - - - - - - - - - - |
                                              |                         v
        Input -> (PReLU -> Conv(5x5) -> BN) -> Maxout -> (PReLU -> Conv (5x5) -> BN) -> Maxout -> (PReLU -> Conv (1x1) -> BN)
              |                              ^
              | - - - - - - - - - - - - - - |
        """
        out0 = torch.maximum(x, self.seq1(x)) if self.is_first else self.seq1(x)
        out1 = torch.maximum(out0, self.seq2(out0))
        out = self.seq3(out1)

        if self.verbose:
            print(f"\nCDB block\n-------")
            print(f"Output shape after the first sequence (BN/PReLU + Conv + BN) -> Maxout: {out0.shape}")
            print(f"Output shape after the second sequence (PReLU + Conv + BN) -> Maxout: {out1.shape}")
            print(f"Output shape after the third sequence (PReLU + Conv + BN): {out.shape}")

        return out


class EncodingCDB(CompetitiveDenseBlock):
    """
    Encoding Competitive Dense Block = CompetitiveDenseBlock + Max Pooling
    """

    def __init__(self, params: dict, is_input=False):
        """
        Constructor
        """
        kernel = params["pool"]
        stride = params["pool_stride"]

        super(EncodingCDB, self).__init__(params=params,
                                                    is_input=False)

        # MaxPool2D: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel,
            stride=stride,
            return_indices=True  # Useful for `torch.nn.MaxUnpool2D`
        )

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Forward method for the encoding CDB

        Returns
        -------
        1) output_tensor
        2) output_block - maxpooled feature map
        3) indices
        """

        print(f"\nEncoding block: {__name__}\n---------")
        output_block = super(EncodingCDB, self).forward(x)
        print(f"Shape after CDB: {output_block.shape}.")
        output_encoder, indices = self.max_pool(output_block)
        print(f"Shape after maxpool: {output_encoder.shape}")
        return output_encoder, output_block, indices


class DecodingCDB(CompetitiveDenseBlock):
    """
    Decoding Competitive Block = Unpool2D + Skip Connection -> Dense Block
    """

    def __init__(self, params: dict):
        """
        Constructor
        """
        super(DecodingCDB, self).__init__(params=params)

        kernel = params["pool"]
        stride = params["pool_stride"]

        self.max_unpool = nn.MaxUnpool2d(
            kernel_size=kernel,
            stride=stride
        )

    def forward(self, x: torch.Tensor, output_block: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """

        print(f"\nDecoding block: {__name__}\n---------")
        unpool_output = self.max_unpool(x, indices)
        print(f"Shape after unpool: {unpool_output.shape}.")
        max_output = torch.maximum(unpool_output, output_block)
        print(f"Shape after maxout: {max_output.shape}.")
        output = super(DecodingCDB, self).forward(max_output)
        print(f"Shape after CDB: {output.shape}")
        return output


class ClassifierBlock(nn.Module):
    """
    The last block in our architecture
    """

    def __init__(self, params: dict):
        """
        Constructor
        """
        super(ClassifierBlock, self).__init__()

        in_channels = params["in_channels"]
        num_classes = params["num_classes"]
        kernel_size = params["classifier_kernel"]
        stride = params["stride"]

        self.conv2d = nn.Conv2d(in_channels=in_channels,
                                out_channels=num_classes,
                                kernel_size=kernel_size,
                                stride=stride)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the prediction probs
        """
        return self.softmax(self.conv2d(x))


# Define the F-CNN architecture
class FCnnModel(nn.Module):
    """
    Model architecture that replicates the FastSurfer F-CNN model.
    This model is based on "Competitive Dense Blocks"

    Attributes
    ----------
    Encoding blocks: enc1-4
    Bottleneck block: bottleneck
    Decoding blocks: dec1-4
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
        params["in_channels"] = filters
        self.enc2 = EncodingCDB(params=params)
        self.enc3 = EncodingCDB(params=params)
        self.enc4 = EncodingCDB(params=params)

        # 2. Bottleneck
        self.bottleneck = CompetitiveDenseBlock(params=params, is_input=False, verbose=True)

        # 3. Defining the decoding sequence:
        self.dec4 = DecodingCDB(params=params)
        self.dec3 = DecodingCDB(params=params)
        self.dec2 = DecodingCDB(params=params)
        self.dec1 = DecodingCDB(params=params)

        # 4. Classifier
        self.classifier = ClassifierBlock(params=params)

        # Initialize the layers:
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes the weights and biases of the U-Net constituent blocks
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Xavier uniform: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.00001)
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
        x4, skip4, ind4 = self.enc2(x3)

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
