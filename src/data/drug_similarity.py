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
