"""
Trainer module for training graph neural networks

This module provides functionality for training GNN models with contrastive loss.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torch.utils.data import DataLoader, TensorDataset

class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning embeddings"""
    
    def __init__(self, margin=0.5):
        """
        Initialize contrastive loss
        
        Args:
            margin: Margin for contrastive loss
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embeddings, pairs):
        """
        Calculate contrastive loss
        
        Args:
            embeddings: Node embeddings
            pairs: Pairs of nodes with labels (0 for negative, 1 for positive)
            
        Returns:
            Loss value
        """
        # Extract pairs and labels
        idx1 = pairs[:, 0]
        idx2 = pairs[:, 1]
        labels = pairs[:, 2].float()
        
        # Get embeddings for pairs
        embed1 = embeddings[idx1]
        embed2 = embeddings[idx2]
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(embed1, embed2, dim=1)
        
        # Transform similarity to be higher for positive pairs and lower for negative pairs
        transformed_sim = 0.5 * (similarity + 1.0)  # Scale from [-1, 1] to [0, 1]
        
        # Calculate contrastive loss
        pos_loss = (1 - labels) * transformed_sim.pow(2)
        neg_loss = labels * F.relu(self.margin - transformed_sim).pow(2)
        
        # Combine losses
        loss = pos_loss + neg_loss
        
        return loss.mean()

class Trainer:
    """Trainer for graph neural networks"""
    
    def __init__(self, model, device, num_epochs=100, batch_size=64, learning_rate=0.001,
                 weight_decay=1e-5, margin=0.5, neg_samples=1, save_model=False):
        """
        Initialize trainer
        
        Args:
            model: GNN model
            device: Device to use for training
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            margin: Margin for contrastive loss
            neg_samples: Number of negative samples per positive sample
            save_model: Whether to save the trained model
        """
        self.model = model
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.margin = margin
        self.neg_samples = neg_samples
        self.save_model = save_model
        
        # Initialize loss function
        self.loss_fn = ContrastiveLoss(margin=margin)
    
    def _generate_training_pairs(self, data, important_targets):
        """
        Generate positive and negative training pairs
        
        Args:
            data: PyG Data object
            important_targets: List of important target indices
            
        Returns:
            Tuple of (positive_pairs, negative_pairs)
        """
        # Extract edge index
        edge_index = data.edge_index.cpu()
        
        # Extract node indices by type
        compound_indices = data.compound_indices.cpu()
        target_indices = data.target_indices.cpu()
        
        # Create set of existing edges for negative sampling
        existing_edges = set()
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            existing_edges.add((src, dst))
            existing_edges.add((dst, src))  # For undirected graph
        
        # Generate positive pairs (existing edges)
        pos_pairs = []
        
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            
            # Only include compound-target pairs for training
            if (src in compound_indices and dst in target_indices) or \
               (src in target_indices and dst in compound_indices):
                # Create pair with label 1 (positive)
                pos_pairs.append([src, dst, 1])
        
        # Generate negative pairs (non-existing edges)
        neg_pairs = []
        
        # Oversample negative pairs if needed
        num_neg_samples = len(pos_pairs) * self.neg_samples
        
        while len(neg_pairs) < num_neg_samples:
            # Randomly sample compounds and targets
            src_idx = np.random.choice(len(compound_indices))
            dst_idx = np.random.choice(len(target_indices))
            
            src = compound_indices[src_idx].item()
            dst = target_indices[dst_idx].item()
            
            # Skip if edge already exists
            if (src, dst) in existing_edges:
                continue
            
            # Create pair with label 0 (negative)
            neg_pairs.append([src, dst, 0])
        
        # Convert to tensors
        pos_pairs = torch.tensor(pos_pairs, dtype=torch.long)
        neg_pairs = torch.tensor(neg_pairs, dtype=torch.long)
        
        print(f"Final counts - Positive pairs: {len(pos_pairs)}, Negative pairs: {len(neg_pairs)}")
        
        return pos_pairs, neg_pairs
    
    def _create_dataloader(self, pos_pairs, neg_pairs):
        """
        Create DataLoader for training pairs
        
        Args:
            pos_pairs: Positive pairs tensor
            neg_pairs: Negative pairs tensor
            
        Returns:
            DataLoader
        """
        # Combine positive and negative pairs
        all_pairs = torch.cat([pos_pairs, neg_pairs], dim=0)
        
        # Shuffle pairs
        idx = torch.randperm(all_pairs.shape[0])
        all_pairs = all_pairs[idx]
        
        # Create dataset
        dataset = TensorDataset(all_pairs)
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        return dataloader
    
    def train(self, data, important_targets):
        """
        Train the model on the given data
        
        Args:
            data: PyG Data object with graph structure
            important_targets: List of important target indices
            
        Returns:
            Node embeddings
        """
        self.model.train()
        
        # Move data to device
        data = data.to(self.device)
        
        # Generate training pairs
        pos_pairs, neg_pairs = self._generate_training_pairs(data, important_targets)
        print(f"Number of positive pairs: {len(pos_pairs)}")
        
        # Create DataLoader for training pairs
        train_loader = self._create_dataloader(pos_pairs, neg_pairs)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # Training loop
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            
            for batch in train_loader:
                batch = batch[0].to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass - making sure to pass edge_type
                embeddings = self.model(data.x, data.edge_index, data.edge_type, data.edge_weight)
                
                # Calculate loss
                loss = self.loss_fn(embeddings, batch)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
                
                total_loss += loss.item()
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {total_loss / len(train_loader):.4f}")
        
        # Generate final embeddings
        self.model.eval()
        with torch.no_grad():
            # Make sure to pass edge_type here as well
            embeddings = self.model(data.x, data.edge_index, data.edge_type, data.edge_weight)
        
        # Save model if specified
        if self.save_model:
            os.makedirs('results/models', exist_ok=True)
            torch.save(self.model.state_dict(), 'results/models/rgcn_model.pt')
            print("Model saved to results/models/rgcn_model.pt")
        
        return embeddings
    
    def plot_training_loss(self):
        """Plot and save the training loss curve"""
        if not self.loss_history:
            print("No loss history to plot")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs('results/visualizations', exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot
        plt.savefig('results/visualizations/training_loss.png', dpi=300)
        plt.close()
        print("Training loss plot saved to results/visualizations/training_loss.png")
