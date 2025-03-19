#!/bin/bash

# Create Enhanced TCM Target Prioritization System
echo "Setting up Enhanced TCM Target Prioritization System..."

# Create directory structure
mkdir -p tcm_target_prioritization
cd tcm_target_prioritization
mkdir -p data/raw data/processed
mkdir -p results/embeddings results/models results/evaluation results/visualizations results/weight_optimization results/prioritization
mkdir -p scripts src/data src/features src/models src/training src/evaluation

# Create requirements.txt
cat > requirements.txt << 'EOF'
torch>=1.8.0
torch-geometric>=2.0.0
numpy>=1.20.0
pandas>=1.2.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
tqdm>=4.60.0
networkx>=2.5.0
rdkit>=2022.03.1    # For chemical structure handling
biopython>=1.79     # For protein sequence handling
EOF

# Create src/data modules
cat > src/data/__init__.py << 'EOF'
# Data package initialization
EOF

cat > src/data/drug_similarity.py << 'EOF'
"""
Drug similarity calculator module

This module provides functions to calculate structural similarity between drugs based on their
molecular fingerprints or SMILES representations.
"""
import os
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from tqdm import tqdm

class DrugSimilarityCalculator:
    """
    Calculator for drug structural similarities
    """
    def __init__(self, fingerprint_type='morgan', radius=2, nBits=2048):
        """
        Initialize drug similarity calculator
        
        Args:
            fingerprint_type: Type of fingerprint to use ('morgan', 'maccs', 'topological')
            radius: Radius for Morgan fingerprints
            nBits: Number of bits for fingerprints
        """
        self.fingerprint_type = fingerprint_type
        self.radius = radius
        self.nBits = nBits
    
    def calculate_fingerprint(self, smiles):
        """
        Calculate molecular fingerprint from SMILES
        
        Args:
            smiles: SMILES representation of molecule
            
        Returns:
            Molecular fingerprint
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        if self.fingerprint_type == 'morgan':
            # Morgan (ECFP) fingerprint
            return AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.nBits)
        elif self.fingerprint_type == 'maccs':
            # MACCS keys (166 bit)
            return MACCSkeys.GenMACCSKeys(mol)
        elif self.fingerprint_type == 'topological':
            # Topological fingerprint
            return Chem.RDKFingerprint(mol, fpSize=self.nBits)
        else:
            # Default to Morgan
            return AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.nBits)
    
    def calculate_similarity_matrix(self, drug_data, smiles_col='smiles', id_col='compound_id'):
        """
        Calculate similarity matrix for all drugs
        
        Args:
            drug_data: DataFrame with drug SMILES
            smiles_col: Name of SMILES column
            id_col: Name of ID column
            
        Returns:
            similarity_df: DataFrame with pairwise similarities
            drug_ids: List of drug IDs
            fingerprints: Dictionary of fingerprints for each drug
        """
        # Extract valid molecules and calculate fingerprints
        fingerprints = {}
        valid_drugs = []
        
        print("Calculating drug fingerprints...")
        for _, row in tqdm(drug_data.iterrows(), total=len(drug_data)):
            drug_id = row[id_col]
            smiles = row[smiles_col]
            
            if isinstance(smiles, str):
                fp = self.calculate_fingerprint(smiles)
                if fp is not None:
                    fingerprints[drug_id] = fp
                    valid_drugs.append(drug_id)
        
        # Create similarity matrix
        n_drugs = len(valid_drugs)
        similarity_matrix = np.zeros((n_drugs, n_drugs))
        
        print("Calculating pairwise similarities...")
        for i in tqdm(range(n_drugs)):
            drug_i_id = valid_drugs[i]
            fp_i = fingerprints[drug_i_id]
            
            # Calculate similarity with all other drugs (including self)
            for j in range(i, n_drugs):
                drug_j_id = valid_drugs[j]
                fp_j = fingerprints[drug_j_id]
                
                # Calculate Tanimoto similarity
                similarity = DataStructs.TanimotoSimilarity(fp_i, fp_j)
                
                # Store in matrix (symmetric)
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        # Convert to DataFrame
        similarity_df = pd.DataFrame(similarity_matrix, index=valid_drugs, columns=valid_drugs)
        
        return similarity_df, valid_drugs, fingerprints
    
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
        print(f"Drug similarity matrix saved to {output_path}")
    
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
        drug_ids = list(similarity_df.columns)
        similarity_df.columns = drug_ids
        
        return similarity_df

def generate_sample_drug_structures(output_path, compounds=None):
    """
    Generate sample drug structure data for testing
    
    Args:
        output_path: Path to save data
        compounds: List of compound names (optional)
    """
    if compounds is None:
        compounds = [
            "Berberine", "Curcumin", "Ginsenoside_Rg1", "Astragaloside_IV", 
            "Baicalein", "Quercetin", "Tanshinone_IIA", "Tetrandrine", 
            "Emodin", "Resveratrol"
        ]
    
    # Common TCM compounds with SMILES
    tcm_smiles = {
        "Berberine": "COc1ccc2c(c1OC)C[n+]3cc4c(cc3C2)OCO4",
        "Curcumin": "COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O",
        "Ginsenoside_Rg1": "CC1(C)CCC2(CCC3(C)C(CCC4C3CCC3(C)C(C(C5OC(CO)C(OC6OC(CO)C(O)C(O)C6O)C(O)C5O)CC43O)C4C)C2C1)O",
        "Astragaloside_IV": "CC1C(C(CC2C(C3C(CC(O3)OC3OC(CO)C(O)C(OC4OC(CO)C(O)C(O)C4O)C3O)CC4(CC(C(C4)C2(C)C)OC2OC(C(C(C2O)O)O)CO)C)C1(C)C)O)O",
        "Baicalein": "O=c1cc(-c2ccccc2)oc2cc(O)c(O)c(O)c12",
        "Quercetin": "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
        "Tanshinone_IIA": "CC1COC2=C1C(=O)C(=O)c1c2ccc2c1CCCC2(C)C",
        "Tetrandrine": "COc1cc2c(cc1OC)C(=O)C1CN(CCc3cc(OC)c(OC)cc31)CC2",
        "Emodin": "Cc1cc(O)c2c(c1)C(=O)c1cc(O)cc(O)c1C2=O",
        "Resveratrol": "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1"
    }
    
    # Create DataFrame
    data = []
    for compound in compounds:
        smiles = tcm_smiles.get(compound, "")
        # If no SMILES available, create a placeholder structure
        if smiles == "":
            # Random SMILES for demonstration
            smiles = "C1CCCCC1" if compound == "Unknown" else f"C1CCCCC1{compound[0]}"
        
        data.append({
            'compound_id': compound,
            'smiles': smiles
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Sample drug structure data saved to {output_path}")
    
    return df
EOF

cat > src/data/protein_similarity.py << 'EOF'
"""
Protein similarity calculator module

This module provides functions to calculate sequence similarity between proteins
using various sequence alignment methods.
"""
import os
import numpy as np
import pandas as pd
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo
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
        
        # Set substitution matrix
        if matrix == 'blosum62':
            self.matrix = MatrixInfo.blosum62
        elif matrix == 'pam250':
            self.matrix = MatrixInfo.pam250
        else:
            self.matrix = MatrixInfo.blosum62
    
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
        
        # Choose alignment method
        if self.method == 'local':
            alignments = pairwise2.align.localds(
                seq1, seq2, self.matrix, self.gap_open, self.gap_extend
            )
        else:  # global
            alignments = pairwise2.align.globalds(
                seq1, seq2, self.matrix, self.gap_open, self.gap_extend
            )
        
        if not alignments:
            return 0.0
        
        # Get best alignment
        best_alignment = alignments[0]
        score = best_alignment.score
        
        # Normalize score by dividing by the maximum possible score (self-alignment)
        seq1_self = pairwise2.align.localds(
            seq1, seq1, self.matrix, self.gap_open, self.gap_extend
        )[0].score
        
        seq2_self = pairwise2.align.localds(
            seq2, seq2, self.matrix, self.gap_open, self.gap_extend
        )[0].score
        
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
EOF

cat > src/data/graph_builder.py << 'EOF'
"""
Graph builder module

This module provides functions to build a graph from knowledge graph data,
incorporating drug and protein similarities.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import json

def build_graph(kg_data, external_data=None, compound_features=None, target_features=None,
                drug_similarity=None, protein_similarity=None,
                drug_sim_threshold=0.5, protein_sim_threshold=0.5):
    """
    Build graph from knowledge graph data and similarity information
    
    Args:
        kg_data: Knowledge graph DataFrame
        external_data: External data for additional edges
        compound_features: Dictionary of compound features
        target_features: Dictionary of target features
        drug_similarity: DataFrame with drug similarity matrix
        protein_similarity: DataFrame with protein similarity matrix
        drug_sim_threshold: Threshold for creating similarity edges between drugs
        protein_sim_threshold: Threshold for creating similarity edges between proteins
        
    Returns:
        data: PyG Data object
        node_map: Mapping from node IDs to indices
        reverse_node_map: Mapping from indices to node IDs
    """
    # Extract all nodes from KG
    compounds = set()
    targets = set()
    
    for _, row in kg_data.iterrows():
        comp = row['compound'] if 'compound' in row else row.get('compound_id', '')
        target = row['target'] if 'target' in row else row.get('target_id', '')
        
        if comp:
            compounds.add(comp)
        if target:
            targets.add(target)
    
    # Create mapping from node names to indices
    all_nodes = list(compounds) + list(targets)
    node_map = {node: i for i, node in enumerate(all_nodes)}
    reverse_node_map = {i: node for node, i in node_map.items()}
    
    # Save mappings as JSON for later use
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/node_map.json', 'w') as f:
        json.dump({str(k): v for k, v in node_map.items()}, f)
    with open('data/processed/reverse_node_map.json', 'w') as f:
        json.dump({str(k): v for k, v in reverse_node_map.items()}, f)
    
    # Prepare edge indices and weights from KG
    src_nodes = []
    dst_nodes = []
    edge_weights = []
    
    for _, row in kg_data.iterrows():
        comp = row['compound'] if 'compound' in row else row.get('compound_id', '')
        target = row['target'] if 'target' in row else row.get('target_id', '')
        confidence = row.get('confidence_score', 1.0)
        
        if comp in node_map and target in node_map:
            # Add edge from compound to target
            src_nodes.append(node_map[comp])
            dst_nodes.append(node_map[target])
            edge_weights.append(float(confidence))
            
            # Add reverse edge (for undirected graph)
            src_nodes.append(node_map[target])
            dst_nodes.append(node_map[comp])
            edge_weights.append(float(confidence))
    
    # Add drug similarity edges if provided
    if drug_similarity is not None:
        print("Adding drug similarity edges...")
        drug_ids = drug_similarity.columns
        
        for i, drug_i in enumerate(drug_ids):
            if drug_i not in node_map:
                continue
                
            for j, drug_j in enumerate(drug_ids):
                if i != j and drug_j in node_map:
                    sim_score = drug_similarity.iloc[i, j]
                    
                    # Only add edges for similar drugs
                    if sim_score >= drug_sim_threshold:
                        # Add similarity edge (bidirectional)
                        src_nodes.append(node_map[drug_i])
                        dst_nodes.append(node_map[drug_j])
                        edge_weights.append(float(sim_score))
                        
                        # No need to add reverse edge since we're looping over all pairs
    
    # Add protein similarity edges if provided
    if protein_similarity is not None:
        print("Adding protein similarity edges...")
        protein_ids = protein_similarity.columns
        
        for i, protein_i in enumerate(protein_ids):
            if protein_i not in node_map:
                continue
                
            for j, protein_j in enumerate(protein_ids):
                if i != j and protein_j in node_map:
                    sim_score = protein_similarity.iloc[i, j]
                    
                    # Only add edges for similar proteins
                    if sim_score >= protein_sim_threshold:
                        # Add similarity edge (bidirectional)
                        src_nodes.append(node_map[protein_i])
                        dst_nodes.append(node_map[protein_j])
                        edge_weights.append(float(sim_score))
                        
                        # No need to add reverse edge since we're looping over all pairs
    
    # Create edge index tensor
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    
    # Create edge weight tensor
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    
    # Extract node indices by type
    compound_indices = torch.tensor([node_map[c] for c in compounds if c in node_map], dtype=torch.long)
    target_indices = torch.tensor([node_map[t] for t in targets if t in node_map], dtype=torch.long)
    
    # Create node features
    node_features = []
    
    # Feature dimension (default if no features provided)
    feature_dim = 128
    if compound_features and len(compound_features) > 0:
        first_feature = next(iter(compound_features.values()))
        feature_dim = first_feature.shape[0]
    
    for i in range(len(all_nodes)):
        node_id = reverse_node_map[i]
        
        if node_id in compounds and node_id in compound_features:
            # Use provided compound features
            node_features.append(compound_features[node_id])
        elif node_id in targets and target_features and node_id in target_features:
            # Use provided target features
            node_features.append(target_features[node_id])
        else:
            # Create random features for nodes without provided features
            node_features.append(torch.randn(feature_dim))
    
    # Stack node features
    x = torch.stack(node_features)
    
    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        compound_indices=compound_indices,
        target_indices=target_indices
    )
    
    print(f"Graph built with {data.x.shape[0]} nodes and {data.edge_index.shape[1]} edges")
    print(f"  - {len(compounds)} compounds")
    print(f"  - {len(targets)} targets")
    
    return data, node_map, reverse_node_map
EOF

# Create calculate_similarities.py script
cat > scripts/calculate_similarities.py << 'EOF'
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
EOF

# Create prepare_data.py script
cat > scripts/prepare_data.py << 'EOF'
import os
import pandas as pd
import numpy as np
import random

def generate_sample_data():
    """Generate sample data files for training"""
    os.makedirs('data/raw', exist_ok=True)
    
    # List of compounds
    compounds = [
        "Berberine", "Curcumin", "Ginsenoside_Rg1", "Astragaloside_IV", 
        "Baicalein", "Quercetin", "Tanshinone_IIA", "Tetrandrine", 
        "Emodin", "Resveratrol"
    ]
    
    # List of targets
    targets = [
        "TNF", "IL1B", "IL6", "NFKB1", "PTGS2", "IL10", "CXCL8", "CCL2",
        "CASP3", "BCL2", "BAX", "TP53", "AKT1", "MAPK1", "JUN", "STAT3"
    ]
    
    # Diseases
    diseases = ["Inflammation", "Cancer", "Diabetes", "Alzheimer"]
    
    # 1. Generate knowledge graph data
    kg_data = []
    for compound in compounds:
        # Each compound connects to multiple targets
        for _ in range(random.randint(3, 8)):
            target = random.choice(targets)
            relation_type = random.choice(['activation', 'inhibition', 'binding'])
            confidence = round(random.uniform(0.6, 0.95), 2)
            kg_data.append({
                'compound': compound,
                'target': target,
                'relation_type': relation_type,
                'confidence_score': confidence
            })
    
    kg_df = pd.DataFrame(kg_data)
    kg_df.to_csv('data/raw/kg_data_extended.csv', index=False)
    print(f"Created knowledge graph data with {len(kg_df)} interactions")
    
    # 2. Generate database data
    db_data = []
    for compound in compounds:
        # Create a random feature vector with 256 elements
        feature_vector = [round(random.uniform(-1, 1), 3) for _ in range(256)]
        db_data.append({
            'compound_id': compound,
            'feature_vector': str(feature_vector)
        })
    
    db_df = pd.DataFrame(db_data)
    db_df.to_csv('data/raw/database_data_extended.csv', index=False)
    print(f"Created database data with {len(db_df)} compounds")
    
    # 3. Generate disease importance data
    importance_data = []
    for disease in diseases:
        for target in targets:
            importance = round(random.uniform(0.3, 0.9), 2)
            importance_data.append({
                'disease': disease,
                'target': target,
                'importance_score': importance
            })
    
    importance_df = pd.DataFrame(importance_data)
    importance_df.to_csv('data/raw/disease_importance_extended.csv', index=False)
    print(f"Created disease importance data with {len(importance_df)} entries")
    
    # 4. Generate validated interactions
    validated_data = []
    for _ in range(20):  # 20 validated interactions
        compound = random.choice(compounds)
        target = random.choice(targets)
        confidence = round(random.uniform(0.7, 0.98), 2)
        validation_method = random.choice(['assay', 'binding', 'literature'])
        
        validated_data.append({
            'compound': compound,
            'target': target,
            'confidence_score': confidence,
            'validation_method': validation_method
        })
    
    validated_df = pd.DataFrame(validated_data)
    validated_df.to_csv('data/raw/validated_interactions.csv', index=False)
    print(f"Created validated interactions with {len(validated_df)} entries")
    
    # 5. Generate drug structures data (simplified SMILES)
    from src.data.drug_similarity import generate_sample_drug_structures
    generate_sample_drug_structures('data/raw/drug_structures.csv', compounds)
    
    # 6. Generate protein sequences data
    from src.data.protein_similarity import generate_sample_protein_sequences
    generate_sample_protein_sequences('data/raw/protein_sequences.csv', targets)
    
    print("Sample data generation complete!")

if __name__ == "__main__":
    generate_sample_data()
EOF

# Create main.py script
cat > scripts/main.py << 'EOF'
#!/usr/bin/env python
"""
Main script for enhanced TCM target prioritization system

This script integrates a graph neural network model with disease importance scores
and similarity information to prioritize potential targets for Traditional Chinese
Medicine compounds.
"""

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.graph_builder import build_graph
from src.models.fixed_graph_sage import FixedGraphSAGE
from src.training.trainer import Trainer
from src.evaluation.validation import validate_model
from src.data.drug_similarity import DrugSimilarityCalculator
from src.data.protein_similarity import ProteinSimilarityCalculator

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced TCM Target Prioritization System')
    
    # Input data options
    parser.add_argument('--kg_data', type=str, default='data/raw/kg_data_extended.csv',
                      help='Knowledge graph data file')
    parser.add_argument('--db_data', type=str, default='data/raw/database_data_extended.csv',
                      help='Database data file')
    parser.add_argument('--disease_importance', type=str, default='data/raw/disease_importance_extended.csv',
                      help='Disease importance data file')
    parser.add_argument('--validated_data', type=str, default='data/raw/validated_interactions.csv',
                      help='Validated interactions file')
    parser.add_argument('--drug_structures', type=str, default='data/raw/drug_structures.csv',
                      help='Drug structures file')
    parser.add_argument('--protein_sequences', type=str, default='data/raw/protein_sequences.csv',
                      help='Protein sequences file')
    parser.add_argument('--drug_similarity', type=str, default='data/processed/drug_similarity.csv',
                      help='Drug similarity matrix file')
    parser.add_argument('--protein_similarity', type=str, default='data/processed/protein_similarity.csv',
                      help='Protein similarity matrix file')
    
    # Similarity options
    parser.add_argument('--use_drug_similarity', action='store_true',
                      help='Use drug structural similarity')
    parser.add_argument('--use_protein_similarity', action='store_true',
                      help='Use protein sequence similarity')
    parser.add_argument('--drug_sim_threshold', type=float, default=0.5,
                      help='Threshold for drug similarity edges')
    parser.add_argument('--protein_sim_threshold', type=float, default=0.5,
                      help='Threshold for protein similarity edges')
    parser.add_argument('--calculate_similarities', action='store_true',
                      help='Calculate similarities if not already done')
    
    # Weight parameters for prioritization
    parser.add_argument('--embedding_weight', type=float, default=0.05,
                      help='Weight for embedding similarity (default: 0.05)')
    parser.add_argument('--importance_weight', type=float, default=0.8,
                      help='Weight for target importance (default: 0.8)')
    parser.add_argument('--drug_sim_weight', type=float, default=0.075,
                      help='Weight for drug similarity (default: 0.075)')
    parser.add_argument('--protein_sim_weight', type=float, default=0.075,
                      help='Weight for protein similarity (default: 0.075)')
    
    # Model parameters
    parser.add_argument('--feature_dim', type=int, default=256,
                      help='Feature dimension (default: 256)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                      help='Hidden layer dimension (default: 256)')
    parser.add_argument('--output_dim', type=int, default=128,
                      help='Output embedding dimension (default: 128)')
    parser.add_argument('--dropout', type=float, default=0.3,
                      help='Dropout probability (default: 0.3)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=300,
                      help='Number of training epochs (default: 300)')
    parser.add_argument('--lr', type=float, default=0.0001,
                      help='Learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='Weight decay (default: 1e-5)')
    parser.add_argument('--margin', type=float, default=0.4,
                      help='Contrastive loss margin (default: 0.4)')
    parser.add_argument('--patience', type=int, default=30,
                      help='Early stopping patience (default: 30)')
    
    # Weight tuning options
    parser.add_argument('--tune_weights', action='store_true',
                      help='Tune weights automatically')
    
    # Evaluation options
    parser.add_argument('--top_k', type=int, default=10,
                      help='Top K targets to return (default: 10)')
    
    # Other options
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Output directory (default: results)')
    parser.add_argument('--save_model', action='store_true',
                      help='Save trained model')
    parser.add_argument('--load_model', type=str, default=None,
                      help='Load model from file path')
    parser.add_argument('--generate_samples', action='store_true',
                      help='Generate sample data if files not found')
    
    return parser.parse_args()

def load_data(args):
    """Load and prepare input data"""
    print("Loading data...")
    
    # Check if files exist
    if not all(os.path.exists(f) for f in [args.kg_data, args.db_data, args.disease_importance]):
        print("Some required files are missing. Checking if we need to generate sample data...")
        if not os.path.exists('data/raw'):
            os.makedirs('data/raw', exist_ok=True)
            print("Generating sample data...")
            from prepare_data import generate_sample_data
            generate_sample_data()
    
    # Generate structure and sequence data if needed
    if args.generate_samples:
        if not os.path.exists(args.drug_structures):
            print("Generating sample drug structures...")
            from src.data.drug_similarity import generate_sample_drug_structures
            generate_sample_drug_structures(args.drug_structures)
        
        if not os.path.exists(args.protein_sequences):
            print("Generating sample protein sequences...")
            from src.data.protein_similarity import generate_sample_protein_sequences
            generate_sample_protein_sequences(args.protein_sequences)
    
    # Load knowledge graph data
    kg_data = pd.read_csv(args.kg_data)
    print(f"Loaded knowledge graph with {len(kg_data)} relations")
    
    # Load compound database data
    db_data = pd.read_csv(args.db_data)
    print(f"Loaded database with {len(db_data)} compounds")
    
    # Load disease importance data
    disease_data = pd.read_csv(args.disease_importance)
    print(f"Loaded disease importance data with {len(disease_data)} entries")
    
    # Load validated interactions if available
    validated_data = None
    if args.validated_data and os.path.exists(args.validated_data):
        validated_data = pd.read_csv(args.validated_data)
        print(f"Loaded {len(validated_data)} validated interactions")
    
    # Load drug similarity matrix if available and requested
    drug_similarity = None
    if args.use_drug_similarity:
        if os.path.exists(args.drug_similarity):
            print(f"Loading drug similarity matrix from {args.drug_similarity}")
            drug_similarity = pd.read_csv(args.drug_similarity, index_col=0)
            # Convert column names to the right type
            drug_similarity.columns = drug_similarity.columns.astype(str)
        elif args.calculate_similarities:
            print("Calculating drug similarities...")
            from scripts.calculate_similarities import calculate_drug_similarities
            drug_similarity = calculate_drug_similarities(args)
        else:
            print(f"Warning: Drug similarity file not found: {args.drug_similarity}")
            print("Use --calculate_similarities to generate it")
    
    # Load protein similarity matrix if available and requested
    protein_similarity = None
    if args.use_protein_similarity:
        if os.path.exists(args.protein_similarity):
            print(f"Loading protein similarity matrix from {args.protein_similarity}")
            protein_similarity = pd.read_csv(args.protein_similarity, index_col=0)
            # Convert column names to the right type
            protein_similarity.columns = protein_similarity.columns.astype(str)
        elif args.calculate_similarities:
            print("Calculating protein similarities...")
            from scripts.calculate_similarities import calculate_protein_similarities
            protein_similarity = calculate_protein_similarities(args)
        else:
            print(f"Warning: Protein similarity file not found: {args.protein_similarity}")
            print("Use --calculate_similarities to generate it")
    
    # Create compound features dictionary
    compound_features = {}
    for _, row in db_data.iterrows():
        comp_id = row.get('compound_id', row.get('compound', f"compound_{_}"))
        try:
            # Parse feature vector string to list of floats
            if 'feature_vector' in row:
                features = eval(row['feature_vector'])
                compound_features[comp_id] = torch.tensor(features, dtype=torch.float32)
            else:
                # Create random features if no feature vector
                compound_features[comp_id] = torch.randn(args.feature_dim, dtype=torch.float32)
        except Exception as e:
            print(f"Error parsing features for {comp_id}: {e}")
            # Create random features as fallback
            compound_features[comp_id] = torch.randn(args.feature_dim, dtype=torch.float32)
    
    # Extract target importance scores
    target_importance = {}
    for _, row in disease_data.iterrows():
        target = row.get('target_id', row.get('target', ''))
        if target:
            importance = row['importance_score']
            # Take maximum importance across diseases
            if target not in target_importance or importance > target_importance[target]:
                target_importance[target] = float(importance)
    
    return kg_data, compound_features, target_importance, validated_data, drug_similarity, protein_similarity

def train_or_load_model(args, data, important_targets):
    """Train model or load pretrained model"""
    # Create model
    model = FixedGraphSAGE(
        in_dim=data.x.shape[1],
        hidden_dim=args.hidden_dim,
        out_dim=args.output_dim,
        dropout=args.dropout
    )
    
    # Load model if specified
    if args.load_model:
        if os.path.exists(args.load_model):
            print(f"Loading model from {args.load_model}")
            model.load_state_dict(torch.load(args.load_model))
            
            # Get embeddings directly
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            data = data.to(device)
            
            model.eval()
            with torch.no_grad():
                embeddings = model(data.x, data.edge_index)
            
            return embeddings.cpu(), model
        else:
            print(f"Model file {args.load_model} not found. Training new model.")
    
    # Train model
    print("Training model...")
    trainer = Trainer(
        model=model,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        margin=args.margin,
        patience=args.patience
    )
    
    embeddings = trainer.train(data, important_targets)
    
    # Save model if requested
    if args.save_model:
        os.makedirs(f'{args.output_dir}/models', exist_ok=True)
        model_path = f'{args.output_dir}/models/graphsage_model.pt'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    # Save embeddings
    os.makedirs(f'{args.output_dir}/embeddings', exist_ok=True)
    torch.save(embeddings, f'{args.output_dir}/embeddings/node_embeddings.pt')
    print(f"Embeddings saved to {args.output_dir}/embeddings/node_embeddings.pt")
    
    return embeddings, model

def calculate_target_priorities_with_similarity(
    embeddings, node_map, reverse_node_map, data,
    target_importance, drug_similarity, protein_similarity,
    embedding_weight=0.05, importance_weight=0.8, 
    drug_sim_weight=0.075, protein_sim_weight=0.075,
    top_k=10):
    """
    Calculate target priorities incorporating similarity information
    
    Args:
        embeddings: Node embeddings
        node_map: Mapping from node IDs to indices
        reverse_node_map: Mapping from indices to node IDs
        data: Graph data object
        target_importance: Dictionary of target importance scores
        drug_similarity: DataFrame with drug similarity matrix
        protein_similarity: DataFrame with protein similarity matrix
        embedding_weight: Weight for embedding scores
        importance_weight: Weight for target importance scores
        drug_sim_weight: Weight for drug similarity scores
        protein_sim_weight: Weight for protein similarity scores
        top_k: Number of top targets to return
        
    Returns:
        Dictionary mapping compounds to their prioritized targets
    """
    print("Calculating target priorities with similarity...")
    
    # Normalize weights to sum to 1.0
    total_weight = embedding_weight + importance_weight + drug_sim_weight + protein_sim_weight
    
    # Apply normalization if sum is not close to 1.0
    if abs(total_weight - 1.0) > 1e-6:
        norm_factor = 1.0 / total_weight
        embedding_weight *= norm_factor
        importance_weight *= norm_factor
        drug_sim_weight *= norm_factor
        protein_sim_weight *= norm_factor
        print(f"Normalized weights: embedding={embedding_weight:.3f}, importance={importance_weight:.3f}, "
             f"drug_sim={drug_sim_weight:.3f}, protein_sim={protein_sim_weight:.3f}")
    
    # Priorities dictionary
    compound_priorities = {}
    
    # Find median importance score for normalization
    median_importance = 0.5
    if target_importance:
        importance_values = list(target_importance.values())
        median_importance = np.median(importance_values)
    
    # Get compound and target indices
    compound_indices = data.compound_indices
    target_indices = data.target_indices
    
    # Calculate priorities for each compound
    for i, compound_idx in enumerate(tqdm(compound_indices, desc="Processing compounds")):
        compound_id = reverse_node_map[compound_idx.item()]
        compound_embedding = embeddings[compound_idx].unsqueeze(0)
        
        # Calculate similarities with all targets
        target_embeddings = embeddings[target_indices]
        embedding_similarities = torch.cosine_similarity(compound_embedding, target_embeddings)
        
        # Create target scores dictionary
        target_scores = {}
        
        for j, target_idx in enumerate(target_indices):
            target_id = reverse_node_map[target_idx.item()]
            embedding_similarity = embedding_similarities[j].item()
            
            # Get importance score for this target
            importance = target_importance.get(target_id, median_importance)
            
            # Apply nonlinear transformation to increase contrast
            adjusted_importance = np.power(importance, 1.5)  # Power transformation
            
            # Initialize similarity scores
            drug_sim_score = 0.0
            protein_sim_score = 0.0
            
            # Add drug structural similarity component if available
            if drug_similarity is not None and compound_id in drug_similarity.columns:
                # Find similar compounds that interact with this target
                interacting_compounds = []
                
                # Look for compounds that interact with this target in the graph
                for other_comp_idx in compound_indices:
                    other_comp_id = reverse_node_map[other_comp_idx.item()]
                    if other_comp_id != compound_id:
                        # Check if there's an edge between other_comp and target
                        # This is a simplified approach - in a real system, you'd check the actual graph structure
                        # Here we're just checking if a random similarity score is above threshold for demonstration
                        sim_score = drug_similarity.loc[compound_id, other_comp_id] if other_comp_id in drug_similarity.columns else 0
                        if sim_score > 0.3:  # Arbitrary threshold
                            interacting_compounds.append((other_comp_id, sim_score))
                
                if interacting_compounds:
                    # Calculate weighted similarity score
                    total_sim = sum(sim for _, sim in interacting_compounds)
                    drug_sim_score = total_sim / len(interacting_compounds)
            
            # Add protein sequence similarity component if available
            if protein_similarity is not None and target_id in protein_similarity.columns:
                # Find similar proteins to this target
                similar_targets = []
                
                for other_target_idx in target_indices:
                    other_target_id = reverse_node_map[other_target_idx.item()]
                    if other_target_id != target_id and other_target_id in protein_similarity.columns:
                        sim_score = protein_similarity.loc[target_id, other_target_id]
                        if sim_score > 0.3:  # Arbitrary threshold
                            similar_targets.append((other_target_id, sim_score))
                
                if similar_targets:
                    # Calculate weighted similarity score based on similar targets
                    total_sim = sum(sim for _, sim in similar_targets)
                    protein_sim_score = total_sim / len(similar_targets)
            
            # Calculate weighted score using all components
            weighted_score = (
                embedding_weight * embedding_similarity + 
                importance_weight * adjusted_importance +
                drug_sim_weight * drug_sim_score +
                protein_sim_weight * protein_sim_score
            )
            
            # Store in scores dictionary
            target_scores[target_id] = weighted_score
        
        # Sort targets by score and keep top k
        sorted_targets = sorted(target_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        compound_priorities[compound_id] = dict(sorted_targets)
    
    return compound_priorities

def save_results(priorities, args):
    """Save prioritization results"""
    # Create output directory
    os.makedirs(f'{args.output_dir}/prioritization', exist_ok=True)
    
    # Convert to format for CSV
    rows = []
    for compound_id, targets in priorities.items():
        for rank, (target_id, score) in enumerate(targets.items(), 1):
            rows.append({
                'compound_id': compound_id,
                'target_id': target_id,
                'rank': rank,
                'priority_score': score
            })
    
    # Create DataFrame and save
    results_df = pd.DataFrame(rows)
    results_path = f'{args.output_dir}/prioritization/target_priorities.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Prioritization results saved to {results_path}")
    
    # Save as JSON for programmatic access
    import json
    json_data = {}
    for compound_id, targets in priorities.items():
        json_data[compound_id] = [
            {'target_id': target_id, 'score': float(score)}
            for target_id, score in targets.items()
        ]
    
    json_path = f'{args.output_dir}/prioritization/target_priorities.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON results saved to {json_path}")

def create_visualization(embeddings, node_map, reverse_node_map, compound_priorities, args):
    """Create visualizations of the results"""
    try:
        from sklearn.manifold import TSNE
        os.makedirs(f'{args.output_dir}/visualizations', exist_ok=True)
        
        # Create a 2D embedding for visualization
        print("Creating t-SNE visualization...")
        
        # Take first 2000 nodes at most to keep visualization manageable
        num_nodes = min(2000, embeddings.shape[0])
        indices = torch.randperm(embeddings.shape[0])[:num_nodes]
        
        # Get embeddings for these nodes
        sample_embeddings = embeddings[indices]
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, num_nodes-1))
        reduced_embeddings = tsne.fit_transform(sample_embeddings.numpy())
        
        # Get node names and types
        node_names = []
        node_types = []
        
        for i, idx in enumerate(indices):
            node_id = reverse_node_map.get(idx.item(), f"Node_{idx.item()}")
            node_names.append(node_id)
            
            # Determine node type based on name
            if isinstance(node_id, str):
                if any(prefix in node_id for prefix in ['TNF', 'IL', 'CASP', 'BCL', 'Target_']):
                    node_types.append('Target')
                elif any(node_id.startswith(p) for p in ['Berberine', 'Curcumin', 'Ginsenoside']):
                    node_types.append('Compound')
                else:
                    node_types.append('Other')
            else:
                node_types.append('Unknown')
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Plot by node type
        for node_type in ['Compound', 'Target', 'Other', 'Unknown']:
            indices = [i for i, t in enumerate(node_types) if t == node_type]
            if indices:
                plt.scatter(
                    reduced_embeddings[indices, 0], 
                    reduced_embeddings[indices, 1],
                    label=node_type,
                    alpha=0.7
                )
        
        plt.title('t-SNE Visualization of Node Embeddings')
        plt.legend()
        plt.savefig(f'{args.output_dir}/visualizations/embeddings_tsne.png', dpi=300)
        print(f"t-SNE visualization saved to {args.output_dir}/visualizations/embeddings_tsne.png")
        
        # Create heatmap for top compounds and targets
        top_compounds = list(compound_priorities.keys())[:5]  # First 5 compounds
        
        if top_compounds:
            # Get all targets for these compounds
            all_targets = set()
            for comp in top_compounds:
                all_targets.update(compound_priorities[comp].keys())
            
            all_targets = list(all_targets)[:10]  # First 10 targets
            
            # Create score matrix
            score_matrix = np.zeros((len(top_compounds), len(all_targets)))
            
            for i, comp in enumerate(top_compounds):
                for j, target in enumerate(all_targets):
                    if target in compound_priorities[comp]:
                        score_matrix[i, j] = compound_priorities[comp][target]
            
            # Create heatmap
            import seaborn as sns
            plt.figure(figsize=(14, 8))
            ax = sns.heatmap(
                score_matrix,
                xticklabels=all_targets,
                yticklabels=top_compounds,
                annot=True,
                cmap='YlGnBu',
                fmt='.3f'
            )
            plt.title('Compound-Target Priority Scores')
            plt.xlabel('Targets')
            plt.ylabel('Compounds')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'{args.output_dir}/visualizations/priority_heatmap.png', dpi=300)
            print(f"Priority heatmap saved to {args.output_dir}/visualizations/priority_heatmap.png")
    
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    kg_data, compound_features, target_importance, validated_data, drug_similarity, protein_similarity = load_data(args)
    
    # Build graph
    print("Building graph...")
    data, node_map, reverse_node_map = build_graph(
        kg_data, None, compound_features, None,
        drug_similarity if args.use_drug_similarity else None,
        protein_similarity if args.use_protein_similarity else None,
        args.drug_sim_threshold, args.protein_sim_threshold
    )
    
    print(f"Graph built with {data.x.shape[0]} nodes and {data.edge_index.shape[1]} edges")
    
    # Extract important targets for training
    important_targets = []
    for target_id, importance in target_importance.items():
        if target_id in node_map:
            target_idx = node_map[target_id]
            if importance > 0.5:  # Only highly important targets
                important_targets.append(target_idx)
    
    print(f"Found {len(important_targets)} important targets for training")
    
    # Train or load model
    embeddings, model = train_or_load_model(args, data, important_targets)
    
    # Calculate target priorities with similarity
    compound_priorities = calculate_target_priorities_with_similarity(
        embeddings, node_map, reverse_node_map, data,
        target_importance, 
        drug_similarity if args.use_drug_similarity else None,
        protein_similarity if args.use_protein_similarity else None,
        args.embedding_weight, args.importance_weight,
        args.drug_sim_weight, args.protein_sim_weight,
        args.top_k
    )
    
    # Save results
    save_results(compound_priorities, args)
    
    # Create visualizations
    create_visualization(embeddings, node_map, reverse_node_map, compound_priorities, args)
    
    # Validate model if validation data is available
    if validated_data is not None:
        print("Validating model against known interactions...")
        # Prepare compound and target lists
        compound_list = [reverse_node_map[idx.item()] for idx in data.compound_indices]
        target_list = [reverse_node_map[idx.item()] for idx in data.target_indices]
        
        # Run validation
        validation_output_path = f'{args.output_dir}/evaluation/validation_results.png'
        metrics = validate_model(
            compound_priorities, validated_data, compound_list, target_list, validation_output_path
        )
        
        # Print metrics
        print("\nModel Validation Results:")
        print(f"  Average Precision: {metrics.get('average_precision', 0):.4f}")
        print(f"  ROC AUC: {metrics.get('roc_auc', 0):.4f}")
        print(f"  Hit@5: {metrics['hit_rates'].get('hit@5', 0):.4f}")
        print(f"  Hit@10: {metrics['hit_rates'].get('hit@10', 0):.4f}")
        print(f"  Hit@20: {metrics['hit_rates'].get('hit@20', 0):.4f}")
    
    print("\nEnhanced TCM target prioritization completed!")
    
    # Print example results
    if compound_priorities:
        print("\nExample Results:")
        example_compound = next(iter(compound_priorities))
        print(f"Top targets for {example_compound}:")
        for target, score in compound_priorities[example_compound].items():
            print(f"  {target}: {score:.4f}")

if __name__ == "__main__":
    main()
EOF

# Create optimize_weights.py script
cat > scripts/optimize_weights.py << 'EOF'
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from itertools import product

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.graph_builder import build_graph
from src.models.fixed_graph_sage import FixedGraphSAGE
from src.training.trainer import Trainer

def load_data():
    """Load and prepare data for training"""
    print("Loading data...")
    
    # Check if files exist, if not generate sample data
    required_files = [
        'data/raw/kg_data_extended.csv',
        'data/raw/database_data_extended.csv',
        'data/raw/disease_importance_extended.csv',
        'data/raw/validated_interactions.csv',
        'data/raw/drug_structures.csv',
        'data/raw/protein_sequences.csv'
    ]
    
    if not all(os.path.exists(f) for f in required_files):
        print("Some required files are missing. Generating sample data...")
        from prepare_data import generate_sample_data
        generate_sample_data()
    
    # Load knowledge graph data
    kg_data = pd.read_csv('data/raw/kg_data_extended.csv')
    print(f"Loaded {len(kg_data)} KG relations")
    
    # Load disease importance data
    disease_data = pd.read_csv('data/raw/disease_importance_extended.csv')
    print(f"Loaded {len(disease_data)} disease-target importance values")
    
    # Load validated data
    validated_data = pd.read_csv('data/raw/validated_interactions.csv')
    print(f"Loaded {len(validated_data)} validated interactions")
    
    # Load drug similarity data if available
    drug_similarity = None
    if os.path.exists('data/processed/drug_similarity.csv'):
        drug_similarity = pd.read_csv('data/processed/drug_similarity.csv', index_col=0)
        drug_similarity.columns = drug_similarity.columns.astype(str)
        print(f"Loaded drug similarity matrix with shape {drug_similarity.shape}")
    
    # Load protein similarity data if available
    protein_similarity = None
    if os.path.exists('data/processed/protein_similarity.csv'):
        protein_similarity = pd.read_csv('data/processed/protein_similarity.csv', index_col=0)
        protein_similarity.columns = protein_similarity.columns.astype(str)
        print(f"Loaded protein similarity matrix with shape {protein_similarity.shape}")
    
    # Extract compound features from database
    db_data = pd.read_csv('data/raw/database_data_extended.csv')
    
    # Create compound features dictionary
    compound_features = {}
    for _, row in db_data.iterrows():
        comp_id = row['compound_id'] if 'compound_id' in row else row.get('compound', f"compound_{_}")
        try:
            # Parse feature vector string to list of floats
            features = eval(row['feature_vector'])
            compound_features[comp_id] = torch.tensor(features, dtype=torch.float32)
        except Exception as e:
            print(f"Error parsing features for {comp_id}: {e}")
            # Create random features as fallback
            compound_features[comp_id] = torch.randn(256, dtype=torch.float32)
    
    # Extract target importance scores
    target_importance = {}
    for _, row in disease_data.iterrows():
        target = row['target_id'] if 'target_id' in row else row['target']
        importance = row['importance_score']
        
        # Store maximum importance across diseases
        if target not in target_importance or importance > target_importance[target]:
            target_importance[target] = float(importance)
    
    return kg_data, compound_features, target_importance, validated_data, drug_similarity, protein_similarity

def train_model_with_embeddings():
    """Train model and get embeddings"""
    # Load data
    kg_data, compound_features, target_importance, _, drug_similarity, protein_similarity = load_data()
    
    # Build graph
    print("Building graph...")
    data, node_map, reverse_node_map = build_graph(
        kg_data, None, compound_features, None,
        drug_similarity, protein_similarity,
        drug_sim_threshold=0.5, protein_sim_threshold=0.5
    )
    
    print(f"Graph built with {data.x.shape[0]} nodes and {data.edge_index.shape[1]} edges")
    
    # Extract important targets
    important_targets = []
    for target_id, importance in target_importance.items():
        if target_id in node_map:
            target_idx = node_map[target_id]
            if importance > 0.5:  # Only highly important targets
                important_targets.append(target_idx)
    
    print(f"Found {len(important_targets)} important targets")
    
    # Initialize model
    model = FixedGraphSAGE(
        in_dim=data.x.shape[1],
        hidden_dim=256,
        out_dim=128,
        dropout=0.3
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        epochs=150,  # Fewer epochs for weight optimization
        lr=0.0001,
        weight_decay=1e-5,
        margin=0.4,
        patience=20
    )
    
    # Train model
    embeddings = trainer.train(data, important_targets)
    
    # Save embeddings
    os.makedirs('results/embeddings', exist_ok=True)
    torch.save(embeddings, 'results/embeddings/node_embeddings.pt')
    
    # Save mappings for later use
    os.makedirs('data/processed', exist_ok=True)
    torch.save(node_map, 'data/processed/node_map.pt')
    torch.save(reverse_node_map, 'data/processed/reverse_node_map.pt')
    
    return embeddings, node_map, reverse_node_map, target_importance, drug_similarity, protein_similarity

def calculate_priorities_with_similarity(
    embeddings, node_map, reverse_node_map, 
    compound_indices, target_indices, 
    target_importance, drug_similarity, protein_similarity,
    embedding_weight, importance_weight, drug_sim_weight, protein_sim_weight):
    """
    Calculate target priorities with specific weight configuration
    
    Args:
        embeddings: Node embeddings
        node_map: Mapping from node IDs to indices
        reverse_node_map: Mapping from indices to node IDs
        compound_indices: Indices of compound nodes
        target_indices: Indices of target nodes
        target_importance: Dictionary of target importance scores
        drug_similarity: Drug similarity matrix
        protein_similarity: Protein similarity matrix
        embedding_weight: Weight for embedding scores
        importance_weight: Weight for importance scores
        drug_sim_weight: Weight for drug similarity
        protein_sim_weight: Weight for protein similarity
        
    Returns:
        Dictionary mapping compounds to prioritized targets
    """
    # Normalize weights to sum to 1.0
    total_weight = embedding_weight + importance_weight + drug_sim_weight + protein_sim_weight
    
    if abs(total_weight - 1.0) > 1e-6:
        norm_factor = 1.0 / total_weight
        embedding_weight *= norm_factor
        importance_weight *= norm_factor
        drug_sim_weight *= norm_factor
        protein_sim_weight *= norm_factor
    
    # Priorities dictionary
    compound_priorities = {}
    
    # Find median importance score for normalization
    median_importance = 0.5
    if target_importance:
        importance_values = list(target_importance.values())
        median_importance = np.median(importance_values)
    
    # Set of all target IDs
    all_target_ids = [reverse_node_map[target_idx.item()] for target_idx in target_indices]
    
    # Calculate priorities for each compound
    for compound_idx in compound_indices:
        compound_id = reverse_node_map[compound_idx.item()]
        compound_embedding = embeddings[compound_idx].unsqueeze(0)
        
        # Calculate similarities with all targets
        target_embeddings = embeddings[target_indices]
        embedding_similarities = torch.cosine_similarity(compound_embedding, target_embeddings)
        
        # Create target scores dictionary
        target_scores = {}
        
        for i, target_idx in enumerate(target_indices):
            target_id = reverse_node_map[target_idx.item()]
            embedding_similarity = embedding_similarities[i].item()
            
            # Get importance score for this target
            importance = target_importance.get(target_id, median_importance)
            adjusted_importance = np.power(importance, 1.5)  # Power transformation
            
            # Initialize similarity scores
            drug_sim_score = 0.0
            protein_sim_score = 0.0
            
            # Add drug structural similarity component if available
            if drug_similarity is not None and compound_id in drug_similarity.columns:
                # Find similar compounds that interact with this target
                interacting_compounds = []
                
                # This is a simplified approach for the optimization script
                # In real implementation, you'd check actual graph connections
                for other_comp_idx in compound_indices:
                    other_comp_id = reverse_node_map[other_comp_idx.item()]
                    if other_comp_id != compound_id and other_comp_id in drug_similarity.columns:
                        sim_score = drug_similarity.loc[compound_id, other_comp_id]
                        if sim_score > 0.3:  # Arbitrary threshold
                            interacting_compounds.append((other_comp_id, sim_score))
                
                if interacting_compounds:
                    # Calculate weighted similarity score
                    total_sim = sum(sim for _, sim in interacting_compounds)
                    drug_sim_score = total_sim / len(interacting_compounds)
            
            # Add protein sequence similarity component if available
            if protein_similarity is not None and target_id in protein_similarity.columns:
                # Find similar proteins to this target
                similar_targets = []
                
                for other_target_id in all_target_ids:
                    if other_target_id != target_id and other_target_id in protein_similarity.columns:
                        sim_score = protein_similarity.loc[target_id, other_target_id]
                        if sim_score > 0.3:  # Arbitrary threshold
                            similar_targets.append((other_target_id, sim_score))
                
                if similar_targets:
                    # Calculate weighted similarity score
                    total_sim = sum(sim for _, sim in similar_targets)
                    protein_sim_score = total_sim / len(similar_targets)
            
            # Calculate weighted score
            weighted_score = (
                embedding_weight * embedding_similarity + 
                importance_weight * adjusted_importance +
                drug_sim_weight * drug_sim_score +
                protein_sim_weight * protein_sim_score
            )
            
            # Store in scores dictionary
            target_scores[target_id] = weighted_score
        
        # Sort targets by score
        sorted_targets = sorted(target_scores.items(), key=lambda x: x[1], reverse=True)
        compound_priorities[compound_id] = dict(sorted_targets)
    
    return compound_priorities

def evaluate_weights(
    embeddings, node_map, reverse_node_map, 
    target_importance, validated_data, drug_similarity, protein_similarity,
    embedding_weight, importance_weight, drug_sim_weight, protein_sim_weight):
    """
    Evaluate performance with a specific weight configuration
    
    Args:
        embeddings: Node embeddings
        node_map: Mapping from node IDs to indices
        reverse_node_map: Mapping from indices to node IDs
        target_importance: Dictionary of target importance scores
        validated_data: DataFrame with validated interactions
        drug_similarity: Drug similarity matrix
        protein_similarity: Protein similarity matrix
        embedding_weight: Weight for embedding similarity
        importance_weight: Weight for target importance
        drug_sim_weight: Weight for drug similarity
        protein_sim_weight: Weight for protein similarity
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Extract compound and target indices
    compound_indices = [idx for id_name, idx
