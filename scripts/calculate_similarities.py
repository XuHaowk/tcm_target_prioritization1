#!/usr/bin/env python
"""
Script to calculate drug and protein similarities

This script calculates structural similarities between drugs and
sequence similarities between proteins, then saves the results.
"""
import os
import sys
import argparse
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.drug_similarity import DrugSimilarityCalculator, generate_sample_drug_structures
from src.data.protein_similarity import ProteinSimilarityCalculator, generate_sample_protein_sequences

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Calculate drug and protein similarities')
    
    # Input data options
    parser.add_argument('--drug_structures', type=str, default='data/raw/drug_structures.csv',
                       help='Drug structures data file')
    parser.add_argument('--protein_sequences', type=str, default='data/raw/protein_sequences.csv',
                       help='Protein sequences data file')
    
    # Output options
    parser.add_argument('--drug_similarity_output', type=str, default='data/processed/drug_similarity.csv',
                       help='Output file for drug similarity matrix')
    parser.add_argument('--protein_similarity_output', type=str, default='data/processed/protein_similarity.csv',
                       help='Output file for protein similarity matrix')
    
    # Calculation options
    parser.add_argument('--drug_fp_type', type=str, default='morgan',
                       help='Drug fingerprint type (morgan, maccs, topological)')
    parser.add_argument('--protein_align_method', type=str, default='local',
                       help='Protein alignment method (local, global)')
    
    # Other options
    parser.add_argument('--generate_samples', action='store_true',
                       help='Generate sample data if input files do not exist')
    
    return parser.parse_args()

def calculate_drug_similarities(args):
    """Calculate and save drug structural similarities"""
    print("\n=== Calculating Drug Structural Similarities ===")
    
    # Check if input file exists
    if not os.path.exists(args.drug_structures):
        if args.generate_samples:
            print(f"Drug structures file not found. Generating sample data.")
            drug_data = generate_sample_drug_structures(args.drug_structures)
        else:
            print(f"Error: Drug structures file not found: {args.drug_structures}")
            return
    else:
        # Load drug structure data
        drug_data = pd.read_csv(args.drug_structures)
    
    print(f"Loaded {len(drug_data)} drug structures.")
    
    # Initialize calculator
    calculator = DrugSimilarityCalculator(fingerprint_type=args.drug_fp_type)
    
    # Calculate similarities
    smiles_col = 'smiles' if 'smiles' in drug_data.columns else 'SMILES'
    id_col = 'compound_id' if 'compound_id' in drug_data.columns else 'compound'
    
    similarity_df, drug_ids, fingerprints = calculator.calculate_similarity_matrix(
        drug_data, smiles_col=smiles_col, id_col=id_col
    )
    
    # Save results
    calculator.save_similarity_matrix(similarity_df, args.drug_similarity_output)
    
    print(f"Calculated similarity matrix for {len(drug_ids)} drugs")
    
    return similarity_df

def calculate_protein_similarities(args):
    """Calculate and save protein sequence similarities"""
    print("\n=== Calculating Protein Sequence Similarities ===")
    
    # Check if input file exists
    if not os.path.exists(args.protein_sequences):
        if args.generate_samples:
            print(f"Protein sequences file not found. Generating sample data.")
            protein_data = generate_sample_protein_sequences(args.protein_sequences)
        else:
            print(f"Error: Protein sequences file not found: {args.protein_sequences}")
            return
    else:
        # Load protein sequence data
        protein_data = pd.read_csv(args.protein_sequences)
    
    print(f"Loaded {len(protein_data)} protein sequences.")
    
    # Initialize calculator
    calculator = ProteinSimilarityCalculator(method=args.protein_align_method)
    
    # Calculate similarities
    sequence_col = 'sequence' if 'sequence' in protein_data.columns else 'SEQUENCE'
    id_col = 'target_id' if 'target_id' in protein_data.columns else 'target'
    
    similarity_df, protein_ids = calculator.calculate_similarity_matrix(
        protein_data, sequence_col=sequence_col, id_col=id_col
    )
    
    # Save results
    calculator.save_similarity_matrix(similarity_df, args.protein_similarity_output)
    
    print(f"Calculated similarity matrix for {len(protein_ids)} proteins")
    
    return similarity_df

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directories
    os.makedirs(os.path.dirname(args.drug_similarity_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.protein_similarity_output), exist_ok=True)
    
    # Calculate drug similarities
    drug_sim = calculate_drug_similarities(args)
    
    # Calculate protein similarities
    protein_sim = calculate_protein_similarities(args)
    
    # Print summary
    print("\n=== Similarity Calculation Summary ===")
    if drug_sim is not None:
        print(f"Drug similarity matrix saved with shape: {drug_sim.shape}")
    if protein_sim is not None:
        print(f"Protein similarity matrix saved with shape: {protein_sim.shape}")
    
    print("\nSimilarity calculation completed!")

if __name__ == "__main__":
    main()
