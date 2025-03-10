"""
Relation-aware graph neural network model

This module provides an R-GCN implementation for handling heterogeneous edges
with different semantic relationship types.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class RGCN(nn.Module):
    """
    Relational Graph Convolutional Network for heterogeneous graphs with edge types
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations, num_bases=None, dropout=0.2):
        """
        Initialize model
        
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output embedding dimension
            num_relations: Number of relation types
            num_bases: Number of basis matrices for R-GCN
            dropout: Dropout probability
        """
        super(RGCN, self).__init__()
        
        # Set number of bases as min(num_relations, hidden_dim) if not specified
        if num_bases is None:
            num_bases = min(num_relations, hidden_dim)
        
        # Define layers
        self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases)
        self.conv3 = RGCNConv(hidden_dim, out_dim, num_relations=num_relations, num_bases=num_bases)
        
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
    
    def forward(self, x, edge_index, edge_type, edge_weight=None):
        """
        Forward pass with relation-aware message passing
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_type: Edge type indices
            edge_weight: Edge weights (not used in RGCNConv)
            
        Returns:
            Node embeddings
        """
        # First R-GCN layer - Note: RGCNConv doesn't use edge_weight
        x = self.conv1(x, edge_index, edge_type)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second R-GCN layer
        x = self.conv2(x, edge_index, edge_type)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Third R-GCN layer
        x = self.conv3(x, edge_index, edge_type)
        
        # Normalize embeddings
        x = F.normalize(x, p=2, dim=1)
        
        return x
