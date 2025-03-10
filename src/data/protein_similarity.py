"""
Protein similarity calculator module

This module provides functions to calculate sequence similarity between proteins
using various sequence alignment methods.
"""
import os
import numpy as np
import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices
from tqdm import tqdm
import random
import string

class ProteinSimilarityCalculator:
    """
    Calculator for protein sequence similarities
    """
    def __init__(self, method='local', matrix='blosum62', gap_open=-10, gap_extend=-0.5):
        """
        Initialize protein similarity calculator
        
        Args:
            method: Alignment method ('local', 'global')
            matrix: Substitution matrix ('blosum62', 'pam250')
            gap_open: Gap opening penalty
            gap_extend: Gap extension penalty
        """
        self.method = method
        self.matrix_name = matrix
        self.gap_open = gap_open
        self.gap_extend = gap_extend
    
    def calculate_sequence_similarity(self, seq1, seq2):
        """
        Calculate similarity between two protein sequences
        
        Args:
            seq1: First protein sequence
            seq2: Second protein sequence
            
        Returns:
            Normalized similarity score
        """
        # Handle edge cases
        if not seq1 or not seq2:
            return 0.0
            
        if seq1 == seq2:
            return 1.0
        
        # Create aligner with appropriate settings
        aligner = Align.PairwiseAligner()
        
        # Set substitution matrix
        if self.matrix_name == 'blosum62':
            aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        elif self.matrix_name == 'pam250':
            aligner.substitution_matrix = substitution_matrices.load("PAM250")
        else:
            aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        
        # Set gap penalties
        aligner.open_gap_score = self.gap_open
        aligner.extend_gap_score = self.gap_extend
        
        # Choose alignment method
        if self.method == 'local':
            aligner.mode = 'local'
        else:  # global
            aligner.mode = 'global'
        
        # Perform alignment
        alignments = aligner.align(seq1, seq2)
        
        if not alignments:
            return 0.0
        
        # Get best alignment
        best_alignment = alignments[0]
        score = best_alignment.score
        
        # Normalize score by dividing by the maximum possible score (self-alignment)
        aligner_self1 = Align.PairwiseAligner()
        aligner_self1.substitution_matrix = aligner.substitution_matrix
        aligner_self1.open_gap_score = self.gap_open
        aligner_self1.extend_gap_score = self.gap_extend
        aligner_self1.mode = aligner.mode
        
        aligner_self2 = Align.PairwiseAligner()
        aligner_self2.substitution_matrix = aligner.substitution_matrix
        aligner_self2.open_gap_score = self.gap_open
        aligner_self2.extend_gap_score = self.gap_extend
        aligner_self2.mode = aligner.mode
        
        seq1_self = aligner_self1.align(seq1, seq1)[0].score
        seq2_self = aligner_self2.align(seq2, seq2)[0].score
        
        # Use minimum of self-scores for normalization
        max_possible = min(seq1_self, seq2_self)
        
        if max_possible == 0:
            return 0.0
            
        normalized_score = score / max_possible
        
        return normalized_score
    
    def calculate_similarity_matrix(self, protein_data, sequence_col='sequence', id_col='target_id'):
        """
        Calculate similarity matrix for all proteins
        
        Args:
            protein_data: DataFrame with protein sequences
            sequence_col: Name of sequence column
            id_col: Name of ID column
            
        Returns:
            similarity_df: DataFrame with pairwise similarities
            protein_ids: List of protein IDs
        """
        # Extract valid proteins
        protein_ids = []
        sequences = {}
        
        print("Processing protein sequences...")
        for _, row in protein_data.iterrows():
            protein_id = row[id_col]
            sequence = row[sequence_col]
            
            if isinstance(sequence, str) and len(sequence) > 0:
                sequences[protein_id] = sequence
                protein_ids.append(protein_id)
        
        # Create similarity matrix
        n_proteins = len(protein_ids)
        similarity_matrix = np.zeros((n_proteins, n_proteins))
        
        print("Calculating pairwise similarities...")
        for i in tqdm(range(n_proteins)):
            protein_i_id = protein_ids[i]
            seq_i = sequences[protein_i_id]
            
            # Self-similarity is 1.0
            similarity_matrix[i, i] = 1.0
            
            # Calculate similarity with other proteins
            for j in range(i+1, n_proteins):
                protein_j_id = protein_ids[j]
                seq_j = sequences[protein_j_id]
                
                # Calculate sequence similarity
                similarity = self.calculate_sequence_similarity(seq_i, seq_j)
                
                # Store in matrix (symmetric)
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        # Convert to DataFrame
        similarity_df = pd.DataFrame(similarity_matrix, index=protein_ids, columns=protein_ids)
        
        return similarity_df, protein_ids
    
    def save_similarity_matrix(self, similarity_df, output_path):
        """
        Save similarity matrix to CSV
        
        Args:
            similarity_df: Similarity DataFrame
            output_path: Path to save CSV
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        similarity_df.to_csv(output_path)
        print(f"Protein similarity matrix saved to {output_path}")
    
    def load_similarity_matrix(self, input_path):
        """
        Load similarity matrix from CSV
        
        Args:
            input_path: Path to load CSV from
            
        Returns:
            Similarity DataFrame
        """
        similarity_df = pd.read_csv(input_path, index_col=0)
        
        # Convert string column names to original format
        protein_ids = list(similarity_df.columns)
        similarity_df.columns = protein_ids
        
        return similarity_df

def generate_sample_protein_sequences(output_path, targets=None):
    """
    Generate sample protein sequence data for testing
    
    Args:
        output_path: Path to save data
        targets: List of target names (optional)
    """
    if targets is None:
        targets = [
            "TNF", "IL1B", "IL6", "NFKB1", "PTGS2", "IL10", "CXCL8", "CCL2",
            "CASP3", "BCL2", "BAX", "TP53", "AKT1", "MAPK1", "JUN", "STAT3"
        ]
    
    # Some real protein sequence segments (first 50 amino acids)
    sample_sequences = {
        "TNF": "MSTESMIRDVELAEEALPKKTGGPQGSRRCLFLSLFSFLIVAGATTLFCLLHFGVIG",
        "IL1B": "MAEVPELASEMMAYYSGNEDDLFFEADGPKQMKCSFQDLDLCPLDGGIQLRISDHHYS",
        "IL6": "MNSFSTSAFGPVAFSLGLLLVLPAAFPAPVPPGEDSKDVAAPHRQPLTSSERIDKQIRY",
        "NFKB1": "MAEDDPYLGRPEQMFHLDPSLTHTIFNPEVFQPQMALPTDGPYLQILEQPKQRGFRFRY",
        "PTGS2": "MLARALLLCAVLALSHTANPCCSHPCQNRGVCMSVGFDQYKCDCTRTGFYGENCTTPEFL",
        "IL10": "MHSSALLCCLVLLTGVRASPGQGTQSENSCTHFPGNLPNMLRDLRDAFSRVKTFFQMKD",
        "CXCL8": "MTSKLAVALLAAFLISAALCEGAVLPRSAKELRCQCIKTYSKPFHPKFIKELRVIESGPH",
        "CCL2": "MKVSAALLCLLLIAATFIPQGLAQPDAINAPVTCCYNFTNRKISVQRLASYRRITSSKCPK",
    }
    
    # Create DataFrame
    data = []
    for target in targets:
        # Use real sequence if available, otherwise generate random sequence
        if target in sample_sequences:
            sequence = sample_sequences[target]
        else:
            # Generate random protein sequence of 50-100 amino acids
            length = random.randint(50, 100)
            amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
            sequence = ''.join(random.choice(amino_acids) for _ in range(length))
        
        data.append({
            'target_id': target,
            'sequence': sequence
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Sample protein sequence data saved to {output_path}")
    
    return df
