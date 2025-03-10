import os
import sys
# Add the parent directory to path so Python can find the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
