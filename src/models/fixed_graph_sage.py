"""
Fixed Graph SAGE model

This module provides a fixed Graph SAGE model for the TCM target prioritization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

class FixedGraphSAGE(nn.Module):
    """Fixed Graph SAGE model"""
    
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
        
        # Store parameters
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout
        
        # Define layers - using GCNConv
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=0.1)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x, edge_index, edge_type=None, edge_weight=None):
        """
        Forward pass
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_type: Edge type indices (not used in this model, included for compatibility)
            edge_weight: Edge weights
            
        Returns:
            Node embeddings
        """
        # First layer
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second layer
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Third layer
        x = self.conv3(x, edge_index, edge_weight)
        
        # Normalize embeddings
        x = F.normalize(x, p=2, dim=1)
        
        return x
