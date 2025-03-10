"""
Fixed GraphSAGE implementation module

This module provides a GraphSAGE model implementation with improved stability for the target
prioritization system.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class FixedGraphSAGE(nn.Module):
    """
    GraphSAGE model with improved stability
    """
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        """
        Initialize model
        
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output embedding dimension
            dropout: Dropout probability
        """
        super(FixedGraphSAGE, self).__init__()
        
        # Define layers
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, out_dim)
        
        # Define dropout
        self.dropout = dropout
        
        # Add batch normalization for stability
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=0.1)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass with proper keyword arguments for edge weights
    
        Args:
            x: Node features
            edge_index: Edge indices
            edge_weight: Edge weights
        
        Returns:
            Node embeddings
        """
        # First GraphSAGE layer
        x = self.conv1(x, edge_index, edge_weight=edge_weight)  # Use keyword argument
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = F.dropout(x, p=self.dropout, training=self.training)
    
        # Second GraphSAGE layer
        x = self.conv2(x, edge_index, edge_weight=edge_weight)  # Use keyword argument
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = F.dropout(x, p=self.dropout, training=self.training)
    
        # Third GraphSAGE layer
        x = self.conv3(x, edge_index, edge_weight=edge_weight)  # Use keyword argument
    
        # Normalize embeddings
        x = F.normalize(x, p=2, dim=1)
    
        return x
