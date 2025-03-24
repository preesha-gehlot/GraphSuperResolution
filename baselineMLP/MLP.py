import torch.nn as nn


class LRtoHRMLP(nn.Module):
    """
    Multi-Layer Perceptron for mapping Low-Resolution to High-Resolution data
    """
    def __init__(self,
                 input_dim: int = 12720,
                 hidden_layers: list = [1024, 2048, 4096],
                 output_dim: int = 35778,
                 dropout_rate: float = 0.3):
        """
        Initialize the MLP architecture for LR to HR mapping
        
        Args:
            input_dim (int): Dimensionality of low-resolution input features
            hidden_layers (list): Number of neurons in hidden layers
            output_dim (int): Dimensionality of high-resolution output features
            dropout_rate (float): Dropout probability for regularization
        """
        super(LRtoHRMLP, self).__init__()
        
        # Create layers dynamically
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Optional: Add output activation 
        # (Note: Sigmoid constrains output to [0,1], which may not be desired for all data)
        layers.append(nn.Sigmoid())
        
        # Use Sequential to create the model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor
        """
        # Directly use the sequential model for forward pass
        return self.model(x)