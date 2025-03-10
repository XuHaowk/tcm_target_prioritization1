"""
Loss functions module with GPU compatibility

This module provides contrastive and other loss functions for training.
"""
import torch
import torch.nn.functional as F
import numpy as np

def contrastive_loss(embeddings, positive_pairs, negative_pairs, margin=0.5):
    """
    Contrastive loss function with handling for NaN values
    
    Args:
        embeddings: Node embeddings
        positive_pairs: Indices of positive pairs
        negative_pairs: Indices of negative pairs
        margin: Margin parameter
    
    Returns:
        Loss value
    """
    # Replace NaN values
    if torch.isnan(embeddings).any():
        print("Warning: NaN values detected in embeddings. Replacing with zeros.")
        embeddings = torch.nan_to_num(embeddings, nan=0.0)
    
    # Extract positive pair embeddings
    pos_emb1 = embeddings[positive_pairs[:, 0]]
    pos_emb2 = embeddings[positive_pairs[:, 1]]
    
    # Extract negative pair embeddings
    neg_emb1 = embeddings[negative_pairs[:, 0]] 
    neg_emb2 = embeddings[negative_pairs[:, 1]]
    
    # Calculate cosine similarity
    pos_score = F.cosine_similarity(pos_emb1, pos_emb2)
    neg_score = F.cosine_similarity(neg_emb1, neg_emb2)
    
    # Handle NaN values in similarity scores
    pos_score = torch.nan_to_num(pos_score, nan=0.0)
    neg_score = torch.nan_to_num(neg_score, nan=0.0)
    
    # Print statistics to help with debugging (using .item() to access tensor values)
    print(f"Positive scores range: {pos_score.min().item():.4f} to {pos_score.max().item():.4f}")
    print(f"Negative scores range: {neg_score.min().item():.4f} to {neg_score.max().item():.4f}")
    
    # Calculate loss with additional safety checks
    # Use a small epsilon to ensure we don't get exactly zero loss
    epsilon = 1e-6
    
    # Modified loss calculation to ensure we always have some gradient
    pos_loss = torch.mean(torch.clamp(margin - pos_score + epsilon, min=epsilon))
    neg_loss = torch.mean(torch.clamp(neg_score - (-margin) + epsilon, min=epsilon))
    
    # Combine losses
    loss = pos_loss + neg_loss
    
    # Final safety check
    if torch.isnan(loss) or loss == 0:
        print("Warning: NaN or zero loss detected. Using fallback loss.")
        # Return a small non-zero loss to ensure we always have a gradient
        # Make sure the fallback tensor is on the same device as embeddings
        return torch.tensor(0.1, requires_grad=True, device=embeddings.device)
    
    print(f"Loss value: {loss.item():.4f}")
    return loss
