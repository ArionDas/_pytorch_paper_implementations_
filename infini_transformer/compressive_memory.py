import torch
from torch import nn

class CompressiveMemory(nn.Module):
    """ This class implements the Compressive Transformer memory module """
    def __init__(self, dim_input: int, dim_key: int, dim_value: int, num_heads: int, segment_len: int, update: str = "linear"):
        """

        Args:
            num_heads (int): number of attention heads
            segment_len (int): segment length (must be a factor of the input sequence length)
            update (str, optional): type of memory update rule to use (default: "linear")
        """
        
        super(CompressiveMemory, self).__init__()
        
        # Input parameters
        self.num_heads = num_heads
        self.segment_size = segment_len
        
        self.dim_input = dim_input
        self.dim_key = dim_key
        self.dim_value = dim_value
        
        self.update = update
        
        # Projections for stacked SDP attention
        self.proj_k = nn.Linear(dim_input, num_heads*dim_key, bias=False)
        self.proj_v = nn.Linear(dim_input, num_heads*dim_value, bias=False)
        self.proj_q = nn.Linear(dim_input, num_heads*dim_key, bias=False)
        
        # Initialize betas for weighted average of dot-product and memory-based attention
        self.betas = nn.Parameter(torch.randn(1, num_heads, 1, dim_value))
        
        # Projection for output
        self.proj_out = nn.Linear(num_heads*dim_value, dim_input, bias=False)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Applieas Scaled Dot-Product Attention to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_input)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim_input)
        """
        batch_size, seq_len, _ = x.shape
        n_seq, rem = divmod(seq_len, self.segment_len)
        
        if rem != 0:
            raise ValueError("sequence length must be divisible by segment length.")
        
        out = []
        
        