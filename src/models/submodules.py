import torch
from torch import nn

# Define the first Competitive Dense Block (CDB)
class CompetitiveDenseBlock(nn.Module):
    """
    Each block consists of three sequences of parametric rectified linear unit (PReLU), convolution (Conv), and
    batch normalization (BN) layers, except for the first encoder block. The first block replaces PReLU with
    BN to normalize the raw inputs.
    """
    def __init__(self, params: dict, is_input=False, verbose=False):
        """
        Initializes the competitive dense block with the given parameters.

        Parameters
        ----------
        params : dict
            Dictionary containing the configuration parameters for the block.
        is_input : bool, optional
            Indicates if this is the first block, by default False.
        verbose : bool, optional
            If True, prints additional debug information, by default False.
        """
        super().__init__()

        self.is_first = is_input
        in_channels = params["in_channels"]
        kernel = params["conv_kernel"]
        filters = params["filters"]
        stride = params["conv_stride"]
        self.verbose = verbose

        # Use padding to ensure the output tensor retains the same dimensions
        padding = ((kernel - 1) // 2, (kernel - 1) // 2)

        # Defining four distinct sequences of layers
        self.seq1 = nn.Sequential(
            nn.BatchNorm2d(in_channels) if is_input else nn.PReLU(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=filters,
                      kernel_size=(kernel, kernel),
                      stride=stride,
                      padding=padding),
            nn.BatchNorm2d(filters)
        )

        self.seq2 = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_channels=filters,
                      out_channels=filters,
                      kernel_size=(kernel, kernel),
                      stride=stride,
                      padding=padding),
            nn.BatchNorm2d(filters)
        )

        self.seq3 = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_channels=filters,
                      out_channels=filters,
                      kernel_size=(kernel, kernel),
                      stride=stride,
                      padding=padding),
            nn.BatchNorm2d(filters)
        )

        self.seq4 = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_channels=filters,
                      out_channels=filters,
                      kernel_size=(kernel, kernel),
                      stride=stride,
                      padding=padding),
            nn.BatchNorm2d(filters)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the competitive dense block.

        Input CDB:
                                            | - - - - - - - - - - - - - - - |
                                           |                               v
        Input -> (BN -> Conv(3x3) -> BN)  -> (PReLU -> Conv (5x5) -> BN) -> Maxout -> (PReLU -> Conv (3x3) -> BN)

        ----------------------------------------------------------------------------------------------------------

        Regular CDB:
                                               | - - - - - - - - - - - - - - - - - - -|
                                              |                                       v
        Input -> (PReLU -> Conv(3x3) -> BN) -> Maxout -> (PReLU -> Conv (3x3) -> BN) -> Maxout -> (PReLU -> Conv (3x3) -> BN)
              |                              ^
              | - - - - - - - - - - - - - - |

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to be processed.

        Returns
        -------
        torch.Tensor
            The output tensor after processing.
        """
        out1 = self.seq1(x) if self.is_first else torch.maximum(x, self.seq1(x))
        out2 = torch.maximum(out1, self.seq2(out1))
        out3 = torch.maximum(out2, self.seq3(out2))
        out4 = self.seq4(out3)

        if self.verbose:
            print(f"\nCDB block\n-------")
            print(f"Output shape after the first sequence (BN/PReLU + Conv + BN) -> Maxout: {out1.shape}")
            print(f"Output shape after the second sequence (PReLU + Conv + BN) -> Maxout: {out2.shape}")
            print(f"Output shape after the third sequence (PReLU + Conv + BN) -> Maxout: {out3.shape}")
            print(f"Output shape after the forth sequence (PReLU + Conv + BN): {out4.shape}")

        return out4


class EncodingCDB(CompetitiveDenseBlock):
    """
    Encoding Competitive Dense Block extends CompetitiveDenseBlock by adding Max Pooling.
    """

    def __init__(self, params: dict, is_input=False):
        """
        Initializes the encoding competitive dense block.

        Parameters
        ----------
        params : dict
            Dictionary containing the configuration parameters.
        is_input : bool, optional
            Indicates if this block is the first in the encoding sequence, by default False.
        """
        kernel = params["pool_kernel"]
        stride = params["pool_stride"]

        super(EncodingCDB, self).__init__(params=params,
                                          is_input=is_input)

        # Define the MaxPool2D layer
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel,
            stride=stride,
            return_indices=True  # Useful for `torch.nn.MaxUnpool2D`
        )

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Forward pass through the encoding block.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to be processed.

        Returns
        -------
        torch.Tensor
            The tensor after max pooling.
        torch.Tensor
            The feature map after the dense block.
        torch.Tensor
            Indices from max pooling for use in unpooling.
        """
        output_block = super(EncodingCDB, self).forward(x)
        output_encoder, indices = self.max_pool(output_block)

        if self.verbose:
            print(f"\nEncoding block: {__name__}\n---------")
            print(f"Shape after CDB: {output_block.shape}.")
            print(f"Shape after maxpool: {output_encoder.shape}")
        return output_encoder, output_block, indices


class DecodingCDB(CompetitiveDenseBlock):
    """
    Decoding Competitive Dense Block adds Unpooling and Skip Connections to CompetitiveDenseBlock.
    """
    def __init__(self, params: dict):
        """
        Initializes the decoding competitive dense block.

        Parameters
        ----------
        params : dict
            Dictionary containing the configuration parameters.
        """
        super(DecodingCDB, self).__init__(params=params)

        self.kernel = params["pool_kernel"]
        self.stride = params["pool_stride"]

    def forward(self, x: torch.Tensor, output_block: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Parameters
        ----------
        x : torch.Tensor
            The input tensor from the previous layer.
        output_block : torch.Tensor
            The feature map from the corresponding encoding block.
        indices : torch.Tensor
            Indices from max pooling to be used for unpooling.

        Returns
        -------
        torch.Tensor
            The output tensor after unpooling and processing through the block.
        """
        max_unpool = nn.MaxUnpool2d(
            kernel_size=self.kernel,
            stride=self.stride
        )
        unpool_output = max_unpool(x, indices, output_size=output_block.shape)
        max_output = torch.maximum(unpool_output, output_block)
        output = super(DecodingCDB, self).forward(max_output)

        if self.verbose:
            print(f"\nDecoding block: {__name__}\n---------")
            print(f"Shape after unpool: {unpool_output.shape}.")
            print(f"Shape after maxout: {max_output.shape}.")
            print(f"Shape after CDB: {output.shape}")
        return output


class ClassifierBlock(nn.Module):
    """
    Defines the final classification block of the network.
    """
    def __init__(self, params: dict):
        """
        Initializes the classifier block.

        Parameters
        ----------
        params : dict
            Dictionary containing the configuration parameters.
        """
        super(ClassifierBlock, self).__init__()

        in_channels = params["in_channels"]
        num_classes = params["num_classes"]
        kernel_size = params["classifier_kernel"]
        stride = params["conv_stride"]

        # Initialize the final convolutional layer
        self.conv2d = nn.Conv2d(in_channels=in_channels,
                                out_channels=num_classes,
                                kernel_size=kernel_size,
                                stride=stride)

        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the prediction probs
        """
        return self.conv2d(x)
        # return self.softmax(self.conv2d(x))