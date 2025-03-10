"""
Feature builder module

This module provides functions to build features for compounds and targets.
"""
import torch
import numpy as np

def build_compound_features(fingerprints, emb_dim=256):
    """
    Build compound features from fingerprints
    
    Args:
        fingerprints: Dictionary of compound fingerprints
        emb_dim: Embedding dimension
        
    Returns:
        Dictionary of compound features
    """
    compound_features = {}
    
    for comp_id, fp in fingerprints.items():
        # Convert fingerprint to numpy array
        fp_array = np.zeros((1,))
        
        # If fingerprint is from RDKit, convert to numpy array
        try:
            fp_array = np.array(fp)
        except:
            # Convert bit vector to numpy array
            array = np.zeros((fp.GetNumBits(),), dtype=np.float32)
            for i in range(fp.GetNumBits()):
                if fp.GetBit(i):
                    array[i] = 1.0
            fp_array = array
        
        # If feature dimension doesn't match embedding dimension, adjust
        if len(fp_array) < emb_dim:
            # Pad with zeros
            padded = np.zeros(emb_dim, dtype=np.float32)
            padded[:len(fp_array)] = fp_array
            fp_array = padded
        elif len(fp_array) > emb_dim:
            # Truncate
            fp_array = fp_array[:emb_dim]
        
        # Convert to torch tensor
        compound_features[comp_id] = torch.tensor(fp_array, dtype=torch.float32)
    
    return compound_features

def build_target_features(sequences, emb_dim=256):
    """
    Build target features from sequences
    
    Args:
        sequences: Dictionary of protein sequences
        emb_dim: Embedding dimension
        
    Returns:
        Dictionary of target features
    """
    # Amino acid encoding
    aa_dict = {
        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
        'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
        'B': 20, 'Z': 21, 'X': 22, 'J': 22, 'O': 22, 'U': 22
    }
    
    target_features = {}
    
    for target_id, seq in sequences.items():
        # Convert sequence to one-hot encoding
        if isinstance(seq, str):
            # Use amino acid counts as a simple feature
            aa_counts = np.zeros(23, dtype=np.float32)  # 20 standard + 3 special AAs
            
            for aa in seq:
                aa_upper = aa.upper()
                if aa_upper in aa_dict:
                    aa_counts[aa_dict[aa_upper]] += 1
            
            # Normalize by sequence length
            if len(seq) > 0:
                aa_counts = aa_counts / len(seq)
            
            # Add sequence length as feature
            features = np.concatenate([aa_counts, [len(seq) / 1000]])  # Normalize length
            
            # Pad or truncate to embedding dimension
            if len(features) < emb_dim:
                padded = np.zeros(emb_dim, dtype=np.float32)
                padded[:len(features)] = features
                features = padded
            elif len(features) > emb_dim:
                features = features[:emb_dim]
            
            # Convert to torch tensor
            target_features[target_id] = torch.tensor(features, dtype=torch.float32)
        else:
            # If no sequence available, use random features
            target_features[target_id] = torch.randn(emb_dim, dtype=torch.float32)
    
    return target_features
