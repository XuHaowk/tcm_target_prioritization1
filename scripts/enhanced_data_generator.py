#!/usr/bin/env python
"""
Enhanced Sample Data Generator for TCM Target Prioritization

This script generates a comprehensive set of sample data for training and testing the
TCM target prioritization model, including more compounds, targets, and relationships.
"""

import os
import sys
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import json

# Add project root to path to allow imports from our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the similarity calculators
from src.data.drug_similarity import DrugSimilarityCalculator
from src.data.protein_similarity import ProteinSimilarityCalculator

def generate_enhanced_sample_data(output_dir='data', sample_size='large'):
    """
    Generate enhanced sample data for TCM target prioritization system
    
    Args:
        output_dir: Base directory for data output
        sample_size: Size of sample data ('small', 'medium', 'large')
    """
    print(f"Generating enhanced sample data (size: {sample_size})...")
    
    # Create directories
    raw_dir = os.path.join(output_dir, 'raw')
    processed_dir = os.path.join(output_dir, 'processed')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Determine sample sizes based on requested size
    if sample_size == 'small':
        num_compounds = 20
        num_targets = 25
        num_diseases = 5
        num_kg_interactions = 100
        num_validated = 30
    elif sample_size == 'medium':
        num_compounds = 50
        num_targets = 60
        num_diseases = 10
        num_kg_interactions = 300
        num_validated = 80
    else:  # large
        num_compounds = 100
        num_targets = 120
        num_diseases = 15
        num_kg_interactions = 800
        num_validated = 150
    
    # 1. Generate compounds list with real TCM compounds
    compounds = generate_compound_list(num_compounds)
    print(f"Created list of {len(compounds)} TCM compounds")
    
    # 2. Generate protein targets list with real targets
    targets = generate_target_list(num_targets)
    print(f"Created list of {len(targets)} protein targets")
    
    # 3. Generate disease list
    diseases = generate_disease_list(num_diseases)
    print(f"Created list of {len(diseases)} diseases")
    
    # 4. Generate knowledge graph data
    kg_data = generate_knowledge_graph(compounds, targets, num_kg_interactions)
    kg_df = pd.DataFrame(kg_data)
    kg_path = os.path.join(raw_dir, 'kg_data_extended.csv')
    kg_df.to_csv(kg_path, index=False)
    print(f"Created knowledge graph with {len(kg_df)} interactions → {kg_path}")
    
    # 5. Generate compound database data with feature vectors
    db_data = generate_compound_database(compounds)
    db_df = pd.DataFrame(db_data)
    db_path = os.path.join(raw_dir, 'database_data_extended.csv')
    db_df.to_csv(db_path, index=False)
    print(f"Created compound database with {len(db_df)} compounds → {db_path}")
    
    # 6. Generate disease importance data
    importance_data = generate_disease_importance(diseases, targets)
    importance_df = pd.DataFrame(importance_data)
    importance_path = os.path.join(raw_dir, 'disease_importance_extended.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"Created disease importance data with {len(importance_df)} entries → {importance_path}")
    
    # 7. Generate validated interactions
    validated_data = generate_validated_interactions(compounds, targets, num_validated)
    validated_df = pd.DataFrame(validated_data)
    validated_path = os.path.join(raw_dir, 'validated_interactions.csv')
    validated_df.to_csv(validated_path, index=False)
    print(f"Created validated interactions with {len(validated_df)} entries → {validated_path}")
    
    # 8. Generate drug structures data with SMILES
    drug_structures = generate_drug_structures(compounds)
    drug_structures_df = pd.DataFrame(drug_structures)
    drug_structures_path = os.path.join(raw_dir, 'drug_structures.csv')
    drug_structures_df.to_csv(drug_structures_path, index=False)
    print(f"Created drug structures data with {len(drug_structures_df)} entries → {drug_structures_path}")
    
    # 9. Generate protein sequences data
    protein_sequences = generate_protein_sequences(targets)
    protein_sequences_df = pd.DataFrame(protein_sequences)
    protein_sequences_path = os.path.join(raw_dir, 'protein_sequences.csv')
    protein_sequences_df.to_csv(protein_sequences_path, index=False)
    print(f"Created protein sequences data with {len(protein_sequences_df)} entries → {protein_sequences_path}")
    
    # 10. Calculate and save drug similarities
    print("Calculating drug structural similarities...")
    calculate_drug_similarities(drug_structures_df, os.path.join(processed_dir, 'drug_similarity.csv'))
    
    # 11. Calculate and save protein similarities
    print("Calculating protein sequence similarities...")
    calculate_protein_similarities(protein_sequences_df, os.path.join(processed_dir, 'protein_similarity.csv'))
    
    # 12. Generate node mappings and save
    print("Generating node mappings...")
    node_map = {}
    reverse_node_map = {}
    
    all_nodes = list(compounds) + list(targets)
    for i, node in enumerate(all_nodes):
        node_map[node] = i
        reverse_node_map[i] = node
    
    with open(os.path.join(processed_dir, 'node_map.json'), 'w') as f:
        json.dump({str(k): v for k, v in node_map.items()}, f, indent=2)
    
    with open(os.path.join(processed_dir, 'reverse_node_map.json'), 'w') as f:
        json.dump({str(k): v for k, v in reverse_node_map.items()}, f, indent=2)
    
    print(f"Node mappings saved with {len(node_map)} entries")
    
    print("\nEnhanced sample data generation complete!")
    return {
        'compounds': compounds,
        'targets': targets,
        'diseases': diseases,
        'kg_interactions': len(kg_data),
        'validated_interactions': len(validated_data)
    }

def generate_compound_list(num_compounds):
    """Generate a comprehensive list of TCM compounds"""
    # Base list of real TCM compounds with their traditional uses
    tcm_compounds = [
        # Commonly used TCM compounds
        "Berberine", "Curcumin", "Ginsenoside_Rg1", "Astragaloside_IV", "Baicalein",
        "Quercetin", "Tanshinone_IIA", "Tetrandrine", "Emodin", "Resveratrol",
        # Additional TCM compounds
        "Artemisinin", "Schisandrin", "Glycyrrhizin", "Paeoniflorin", "Silymarin",
        "Ginkgolide_B", "Magnolol", "Honokiol", "Shikonin", "Saikosaponin_A",
        "Matrine", "Oxymatrine", "Andrographolide", "Salvianolic_acid_B", "Triptolide",
        "Puerarin", "Apigenin", "Kaempferol", "Icariin", "Wogonin",
        "Luteolin", "Catechin", "Epicatechin", "Ginsenoside_Rb1", "Gastrodin",
        "Vanillic_acid", "Ferulic_acid", "Crocin", "Curcuminoid", "Epigallocatechin",
        "Capsaicin", "Dioscin", "Platycodin_D", "Phillyrin", "Rotundine",
        "Hypericin", "Imperatorin", "Jujuboside_A", "Ligustilide", "Ginsenoside_Re",
        # More TCM compounds for large dataset
        "Formononetin", "Biochanin_A", "Asiaticoside", "Madecassoside", "Forsythiaside",
        "Arctigenin", "Glabridin", "Liquiritin", "Genistein", "Psoralen",
        "Osthole", "Praeruptorin_A", "Rhein", "Aloe_emodin", "Gallic_acid",
        "Phillygenin", "Bufalin", "Notoginsenoside_R1", "Polydatin", "Luteoloside",
        "Ginsenoside_Rd", "Isoflavone", "Limonin", "Loganin", "Morroniside",
        "Naringenin", "Oridonin", "Patchouli_alcohol", "Rutin", "Salidroside",
        "Sarsasapogenin", "Scutellarin", "Tectorigenin", "Ursolic_acid", "Verbascoside",
        "Celastrol", "Allicin", "Elemicin", "Eugenol", "Cinnamaldehyde",
        "Menthol", "Thymol", "Bergenin", "Daidzein", "Gingerol",
        "Harpagoside", "Hesperidin", "Isorhamnentin", "Kirenol", "Lycorine"
    ]
    
    # If we need more compounds than in our base list, generate some with numbered identifiers
    if num_compounds > len(tcm_compounds):
        additional_needed = num_compounds - len(tcm_compounds)
        for i in range(additional_needed):
            tcm_compounds.append(f"TCM_Compound_{i+1}")
    
    # Take the required number of compounds
    compounds = tcm_compounds[:num_compounds]
    
    # Shuffle to randomize
    random.shuffle(compounds)
    
    return compounds

def generate_target_list(num_targets):
    """Generate a comprehensive list of protein targets relevant to TCM"""
    # Base list of real protein targets related to diseases treated by TCM
    protein_targets = [
        # Inflammation related
        "TNF", "IL1B", "IL6", "NFKB1", "PTGS2", "IL10", "CXCL8", "CCL2",
        # Apoptosis related
        "CASP3", "BCL2", "BAX", "TP53", "CASP9", "CASP8", "PARP1", "CYCS",
        # Signaling pathways
        "AKT1", "MAPK1", "JUN", "STAT3", "MAPK3", "MAPK8", "MAPK14", "PIK3CA",
        # Oxidative stress
        "SOD1", "CAT", "GPX1", "NOS2", "HMOX1", "NFE2L2", "NQO1", "GSR",
        # Metabolism
        "INSR", "SLC2A4", "PPARG", "PPARA", "PCK1", "G6PC", "ACACA", "FASN",
        # Neurotransmission
        "MAOA", "MAOB", "SLC6A4", "SLC6A2", "SLC6A3", "ACHE", "CHRNA7", "HTR2A",
        # Cancer related
        "EGFR", "ERBB2", "MYC", "VEGFA", "CCND1", "CDK4", "RB1", "MDM2",
        # Cardiovascular
        "ACE", "NOS3", "AGTR1", "ADRB1", "ADRB2", "ADRA1A", "EDNRA", "KCNH2",
        # Immune system
        "CD4", "CD8A", "IFNG", "IL4", "IL17A", "TGFB1", "FOXP3", "CD40LG",
        # Other important targets
        "ESR1", "AR", "PGR", "HDAC1", "DNMT1", "HIF1A", "MTOR", "SIRT1",
        # More targets for large dataset
        "PTEN", "KRAS", "BRAF", "PIK3R1", "JAK2", "CXCR4", "CDK2", "ATM",
        "CTNNB1", "NOTCH1", "PRKACA", "GSK3B", "HDAC2", "HDAC3", "DAPK1", "CASP7",
        "MMP9", "MMP2", "TIMP1", "PLAU", "SERPINE1", "ICAM1", "VCAM1", "PTPN1",
        "IDO1", "GSTM1", "GSTT1", "CYP1A1", "CYP3A4", "CYP2D6", "ABCB1", "SLC6A2",
        "DRD2", "HTR1A", "GABRA1", "GRIN1", "OPRM1", "CNR1", "TRPV1", "IL13",
        "IL5", "CCR5", "TLR4", "TLR2", "NLRP3", "PTPRC", "CD3E", "CD19",
        "CD86", "CD80", "CTLA4", "PDCD1", "CD274", "ITGAL", "ITGB2", "SELP"
    ]
    
    # If we need more targets than in our base list, generate some with numbered identifiers
    if num_targets > len(protein_targets):
        additional_needed = num_targets - len(protein_targets)
        for i in range(additional_needed):
            protein_targets.append(f"Target_{i+1}")
    
    # Take the required number of targets
    targets = protein_targets[:num_targets]
    
    # Shuffle to randomize
    random.shuffle(targets)
    
    return targets

def generate_disease_list(num_diseases):
    """Generate list of diseases relevant to TCM"""
    # Base list of diseases commonly treated with TCM
    all_diseases = [
        # Common categories
        "Inflammation", "Cancer", "Diabetes", "Alzheimer", "Cardiovascular_Disease",
        "Arthritis", "Asthma", "Depression", "Parkinson", "Hypertension",
        "Liver_Fibrosis", "Obesity", "Osteoporosis", "Insomnia", "Gastritis",
        # More specific diseases
        "Rheumatoid_Arthritis", "Type_2_Diabetes", "Colorectal_Cancer", "Breast_Cancer",
        "Chronic_Obstructive_Pulmonary_Disease", "Chronic_Kidney_Disease", "Atherosclerosis",
        "Non_Alcoholic_Fatty_Liver_Disease", "Irritable_Bowel_Syndrome", "Coronary_Heart_Disease",
        "Cerebral_Ischemia", "Multiple_Sclerosis", "Ulcerative_Colitis", "Crohn_Disease",
        "Allergic_Rhinitis", "Eczema", "Psoriasis", "Amyotrophic_Lateral_Sclerosis",
        "Epilepsy", "Anxiety", "Bipolar_Disorder", "Schizophrenia", "Migraine",
        "Dysmenorrhea", "Prostate_Cancer", "Lung_Cancer", "Hepatocellular_Carcinoma"
    ]
    
    # Take the required number of diseases
    if num_diseases > len(all_diseases):
        # Generate additional diseases if needed
        additional_needed = num_diseases - len(all_diseases)
        for i in range(additional_needed):
            all_diseases.append(f"Disease_{i+1}")
    
    diseases = all_diseases[:num_diseases]
    
    # Shuffle to randomize
    random.shuffle(diseases)
    
    return diseases

def generate_knowledge_graph(compounds, targets, num_interactions):
    """Generate knowledge graph data connecting compounds and targets"""
    # Calculate number of interactions to create (may be less than num_interactions
    # if we can't create enough unique ones)
    max_interactions = len(compounds) * len(targets)
    interactions_to_create = min(num_interactions, max_interactions)
    
    # Create interactions set to ensure uniqueness
    interactions = set()
    relation_types = ['activation', 'inhibition', 'binding', 'substrate', 'cofactor', 'allosteric_modulator']
    
    # Generate a specific number of interactions
    while len(interactions) < interactions_to_create:
        compound = random.choice(compounds)
        target = random.choice(targets)
        relation_type = random.choice(relation_types)
        
        # Create unique interaction key
        interaction_key = (compound, target)
        
        # Only add if this pair doesn't already exist
        if interaction_key not in interactions:
            interactions.add(interaction_key)
    
    # Convert to list of dictionaries
    kg_data = []
    for compound, target in interactions:
        # Generate confidence score using a beta distribution for more realistic spread
        # Higher alpha parameter (3) gives more high confidence scores (common in biomedical literature)
        confidence = round(np.random.beta(3, 2), 3)
        
        # Determine relation type with probabilities
        # More inhibition relationships than others (common in drug-target interactions)
        relation_probs = {
            'inhibition': 0.6,      # Most drug interactions are inhibitions
            'activation': 0.15,     # Activation still common
            'binding': 0.15,        # Non-specific binding
            'substrate': 0.05,      # For enzymes
            'cofactor': 0.02,       # Rare
            'allosteric_modulator': 0.03  # Rare but important
        }
        
        relation_type = random.choices(
            list(relation_probs.keys()),
            weights=list(relation_probs.values()),
            k=1
        )[0]
        
        kg_data.append({
            'compound': compound,
            'target': target,
            'relation_type': relation_type,
            'confidence_score': confidence
        })
    
    # Add a few high-confidence benchmark interactions
    benchmark_compounds = compounds[:3] if len(compounds) >= 3 else compounds
    benchmark_targets = targets[:3] if len(targets) >= 3 else targets
    
    for i in range(min(len(benchmark_compounds), len(benchmark_targets))):
        # Add a high confidence interaction that we know should be prioritized
        kg_data.append({
            'compound': benchmark_compounds[i],
            'target': benchmark_targets[i],
            'relation_type': 'inhibition',
            'confidence_score': 0.95 + random.uniform(0, 0.05)  # Very high confidence
        })
    
    return kg_data

def generate_compound_database(compounds):
    """Generate compound database with feature vectors"""
    # Create a feature vector for each compound
    db_data = []
    
    # Feature dimensions
    feature_dim = 256  # 256-dimensional feature vectors
    
    for compound in compounds:
        # Generate a feature vector that has some structure (not completely random)
        # We'll create vectors with different distributions based on compound name first letter
        # to simulate chemical similarity clustering
        
        # Create categories based on first letter
        first_letter = compound[0].lower()
        category = ord(first_letter) % 5  # 5 different feature distributions
        
        if category == 0:
            # Category 0: Standard normal distribution
            feature_vector = np.random.normal(0, 1, feature_dim)
        elif category == 1:
            # Category 1: Bimodal distribution
            if random.random() < 0.5:
                feature_vector = np.random.normal(-2, 0.5, feature_dim)
            else:
                feature_vector = np.random.normal(2, 0.5, feature_dim)
        elif category == 2:
            # Category 2: Uniform distribution
            feature_vector = np.random.uniform(-1, 1, feature_dim)
        elif category == 3:
            # Category 3: Mostly positive features
            feature_vector = np.abs(np.random.normal(0, 1, feature_dim))
        else:
            # Category 4: Mostly negative features
            feature_vector = -np.abs(np.random.normal(0, 1, feature_dim))
        
        # Round to 3 decimal places for readability
        feature_vector = np.round(feature_vector, 3).tolist()
        
        # Add to database
        db_data.append({
            'compound_id': compound,
            'feature_vector': str(feature_vector)
        })
    
    return db_data

def generate_disease_importance(diseases, targets):
    """Generate disease importance data for targets"""
    importance_data = []
    
    # Fixed category assignments for certain targets
    target_categories = {
        # Inflammation targets
        'TNF': 'inflammation',
        'IL1B': 'inflammation',
        'IL6': 'inflammation',
        'NFKB1': 'inflammation',
        'PTGS2': 'inflammation',
        'IL10': 'inflammation',
        
        # Apoptosis targets
        'CASP3': 'apoptosis',
        'BCL2': 'apoptosis',
        'BAX': 'apoptosis',
        'TP53': 'apoptosis',
        
        # Signaling targets
        'AKT1': 'signaling',
        'MAPK1': 'signaling',
        'JUN': 'signaling',
        'STAT3': 'signaling',
        
        # Default for others
        'default': 'unknown'
    }
    
    # Fixed disease categories
    disease_categories = {
        'Inflammation': ['inflammation', 'signaling'],
        'Arthritis': ['inflammation'],
        'Rheumatoid_Arthritis': ['inflammation'],
        'Asthma': ['inflammation'],
        'Cancer': ['apoptosis', 'signaling'],
        'Colorectal_Cancer': ['apoptosis', 'signaling'],
        'Breast_Cancer': ['apoptosis', 'signaling'],
        'Alzheimer': ['apoptosis', 'signaling'],
        'Parkinson': ['apoptosis'],
        'Diabetes': ['signaling'],
        'Type_2_Diabetes': ['signaling'],
        'Cardiovascular_Disease': ['signaling', 'inflammation'],
        'Hypertension': ['signaling']
    }
    
    # For each disease-target pair, generate importance score
    for disease in diseases:
        # Get preferred categories for this disease
        preferred_categories = disease_categories.get(disease, ['unknown'])
        
        for target in targets:
            # Get category for this target
            target_category = target_categories.get(target, target_categories['default'])
            
            # Set base importance based on category match
            if target_category in preferred_categories:
                # Higher importance for matching categories
                base_importance = random.uniform(0.65, 0.95)
            else:
                # Lower importance for non-matching categories
                base_importance = random.uniform(0.30, 0.65)
            
            # Add some randomness for realistic variation
            importance = round(min(0.98, base_importance + random.uniform(-0.1, 0.1)), 3)
            
            importance_data.append({
                'disease': disease,
                'target': target,
                'importance_score': importance
            })
    
    return importance_data

def generate_validated_interactions(compounds, targets, num_validated):
    """Generate validated interaction data"""
    # Create validated interactions that partially overlap with KG data
    validated_data = []
    
    # Create a set of unique compound-target pairs
    validated_pairs = set()
    
    # Methods of validation with their prevalence
    validation_methods = {
        'in_vitro_assay': 0.4,
        'in_vivo_study': 0.2,
        'binding_assay': 0.15,
        'clinical_trial': 0.05,
        'literature': 0.2
    }
    
    # Generate a specific number of validated interactions
    while len(validated_pairs) < num_validated and len(validated_pairs) < len(compounds) * len(targets):
        compound = random.choice(compounds)
        target = random.choice(targets)
        
        # Create unique interaction key
        interaction_key = (compound, target)
        
        # Only add if this pair doesn't already exist
        if interaction_key not in validated_pairs:
            validated_pairs.add(interaction_key)
            
            # Generate confidence score - higher on average since these are "validated"
            confidence = round(np.random.beta(4, 1.5), 3)  # Beta distribution favoring higher values
            
            # Select validation method based on probabilities
            validation_method = random.choices(
                list(validation_methods.keys()),
                weights=list(validation_methods.values()),
                k=1
            )[0]
            
            validated_data.append({
                'compound': compound,
                'target': target,
                'confidence_score': confidence,
                'validation_method': validation_method
            })
    
    return validated_data

def generate_drug_structures(compounds):
    """Generate drug structure data with SMILES notations"""
    # Real SMILES for common TCM compounds
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
        "Resveratrol": "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1",
        "Artemisinin": "CC1CCC2C(C)C(OO2)C3C(C)C(C=O)C1C3",
        "Schisandrin": "COc1cc2c(cc1OC)OCO2.COc1cc2c(cc1OC)OCO2",
        "Glycyrrhizin": "CC1(C)CCC2(CCC3(C)C(CCC4C3CCC3(C)C(C(CCC(=O)O)CC34O)C4C)C2C1)OC1OC(C(=O)O)C(OC2OC(C(=O)O)C(O)C(O)C2O)C(O)C1O",
        "Paeoniflorin": "CC1C(C(C(O1)OC1C(C(C(C(O1)CO)O)O)O)OC(=O)C=CC1=CC=C(C=C1)O)O",
        "Silymarin": "COc1cc(C2Oc3cc(O)cc(O)c3C(=O)C2c2ccc(O)c(O)c2)ccc1O",
        "Ginkgolide_B": "CC12CCC(=O)C=C1C(C(C(=O)OC)(C(C1C(C(C(=O)OC)(O1)O2)O)O)O)C",
        "Magnolol": "Oc1ccc(Cc2ccccc2O)cc1",
        "Honokiol": "Oc1ccc(Cc2cc(O)ccc2)cc1",
        "Shikonin": "CC(C)=CCC1=C(O)C(=O)c2c(O)ccc(O)c2C1=O",
        "Saikosaponin_A": "CC1CCC2(C)CCC3(C)C(=CCC4C5(C)CCC(O)C(C)(C)C5CCC43C)C2C1C",
        "Matrine": "CN1CCC2CC(=O)N3CCCC(CC1)C23",
        "Oxymatrine": "CN1CCC2CC(=O)N3CCCC(CC1)C23=O",
        "Andrographolide": "CC1CCC2(CCC3(C)C(CCC4C3C(C=C)CC4(C)C)C2C1C=O)CO",
        "Salvianolic_acid_B": "O=C(O)C=Cc1ccc(O)c(O)c1.O=C(O)C=Cc1ccc(O)c(O)c1.O=C(O)C=Cc1ccc(O)c(O)c1.O=C(O)C=Cc1ccc(O)c(O)c1",
        "Triptolide": "CC1(C)C(CCC2(C)C1CCC1(C)C2C(=O)OC2OC(=O)C3OC23C1)OC(=O)C=C",
        "Puerarin": "O=c1cc(-c2ccc(O)cc2)c2cc(O)c(O)c(O)c2o1",
        "Apigenin": "O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12",
        "Kaempferol": "O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12",
        "Icariin": "CC1OC(OC2C(COC3OC(C)C(O)C(O)C3O)OC(OC3=CC=C4C(OC5=C(O)C=CC(=C5)OC)CCC4C3)C(O)C2O)C(O)C(O)C1O",
        "Wogonin": "COc1c(O)cc(O)c2c(=O)cc(-c3ccccc3)oc12",
        "Luteolin": "O=c1cc(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
        "Catechin": "OC1=CC=C(C=C1)[C@@H]1OC2=C(O)C=C(O)C=C2[C@@H]1O",
        "Epicatechin": "OC1=CC=C(C=C1)[C@@H]1OC2=C(O)C=C(O)C=C2[C@H]1O",
        "Ginsenoside_Rb1": "CC1(C)CCC2(CCC3(C)C(CCC4C3CCC3(C)C(C(C(=O)O)CC43O)C4C)C2C1)OC1OC(C(=O)O)C(OC2OC(C(=O)O)C(O)C(O)C2O)C(O)C1O",
        "Gastrodin": "OCC1OC(OC2=CC=CC=C2CO)C(O)C(O)C1O",
        "Vanillic_acid": "COc1cc(C(=O)O)ccc1O",
        "Ferulic_acid": "COc1cc(/C=C/C(=O)O)ccc1O",
        "Crocin": "CC=CC(=O)OCC1OC(OC2C(O)C(OC=CC(=O)OC)C(O)C(CO)O2)C(O)C(O)C1O"
    }
    
    # Simplified fragment SMILES for generating variations
    fragments = [
        "c1ccccc1", "C=CC=O", "C1CCCCC1", "c1ccc(O)cc1",
        "COc1ccccc1", "Cc1ccccc1", "OC1CCCCC1", "C1=COC=C1",
        "C(=O)O", "c1ccncc1", "C=CC=C", "c1cc(C)ccc1"
    ]
    
    # Create structure data
    structures = []
    
    for compound in compounds:
        # Use real SMILES if available, otherwise generate a synthetic one
        if compound in tcm_smiles:
            smiles = tcm_smiles[compound]
        else:
            # Generate a synthetic SMILES by combining fragments
            # This is a very simplified approach - real SMILES would be more complex
            num_fragments = random.randint(2, 4)
            selected_fragments = random.sample(fragments, num_fragments)
            
            # Create a simple SMILES by joining fragments
            smiles = '.'.join(selected_fragments)
            
            # Add some random decorations to make compounds more diverse
            if random.random() < 0.5:
                smiles += "O"
            if random.random() < 0.3:
                smiles += "N"
        
        structures.append({
            'compound_id': compound,
            'smiles': smiles
        })
    
    return structures

def generate_protein_sequences(targets):
    """Generate protein sequence data"""
    # Real partial protein sequences (first 60-80 amino acids) for common targets
    target_sequences = {
        "TNF": "MSTESMIRDVELAEEALPKKTGGPQGSRRCLFLSLFSFLIVAGATTLFCLLHFGVIGPQREEFPRDLSLISPLAQAVRSSSRTPSDKPVAHVVANPQAEGQLQWLNRRANALLANGVELRDNQLVVPSEGLYLIYSQVLFKGQGCPSTHVLLTHTISRIAVSYQTKVNLLSAIKSPCQRETPEGAEAKPWYEPIYLGGVFQLEKGDRLSAEINRPDYLDFAESGQVYFGIIAL",
        "IL1B": "MAEVPELASEMMAYYSGNEDDLFFEADGPKQMKCSFQDLDLCPLDGGIQLRISDHHYSKGFRQAASVVVAMDKLRKMLVPCPQTFQENDLSTFFPFIFEEEPIFFDTWDNEAYVHDAPVRSLNCTLRDSQQKSLVMSGPYELKALHLQGQDMEQQVVFSMSFVQGEESNDKIPVALGLKEKNLYLSCVLKDDKPTLQLESVDPKNYPKKKMEKRFVFNKIEINNKLEFESAQFPNWYISTSQAENMPVFLGGTKGGQDITDFTMQFVSS",
        "IL6": "MNSFSTSAFGPVAFSLGLLLVLPAAFPAPVPPGEDSKDVAAPHRQPLTSSERIDKQIRYILDGISALRKETCNKSNMCESSKEALAENNLNLPKMAEKDGCFQSGFNEETCLVKIITGLLEFEVYLEYLQNRFESSEEQARAVQMSTKVLIQFLQKKAKNLDAITTPDPTTNASLLTKLQAQNQWLQDMTTHLILRSFKEFLQSSLRALRQM",
        "NFKB1": "MAEDDPYLGRPEQMFHLDPSLTHTIFNPEVFQPQMALPTDGPYLQILEQPKQRGFRFRYVCEGPSHGGLPGASSEKNKKSYPQVKICNYVGPAKVIVQLVTNGKNIHLHAHSLVGKHCEDGICTVTAGPKDMVVGFANLGILHVTKKKVFETLEARMTEACIRGYNPGLLVHSDLAYLQAEGGGDRQLGDREKELIRQAALQQTKEMDLSVVRLMFTAFLPDSTGSFTRRLEPVVSDAIYDSKAPNASNLKIVRMDRTAGCVTGGEEIYLLCDKVQKDDIQIRFYEEEENGGVWEGFGDFSPTDVHRQFAIVFKTPKYKDINITKPASVFVQLRRKSDLETSEPKPFLYYPEIKDKEEVQRKRQKLMPNFSDSFGGGSGAGAGGGGMFGSGGGGGGTGSTGPGYSFPHYGFPTYGGITFHPGTTKSNAGMKHGTMDTESKKDPEGCDKSDDKNTVNLFGKVIETTEQDQEPSEATVGNGEVTLTYATGTKEESAGVQDNLFLEKAMQLAKRHANALFDYAVTGDVKMLLAVQRHLTAVQDENGDSVLHLAIIHLHSQLVRDLLEVTSGLISDDIINMRNDLYQTPLHLAVITKQEDVVEDLLRAGADLSLLDRLGNSVLHLAAKEGHDKVLSILLKHKKAALLLDHPNGDGLNAIHLAMMSNSLPCLLLLVAAGADVNAQEQKSGRTALHLAVEHDNISLAGCLLLEGDAHVDSTTYDGTTPLHIAAGRGGLDTLRILLKHGADVNVPDGTGCTPLHLAAKYGNENVKLLVKAGASVFVSSSEYKNTPLHLAAMEGHDEIVKALLKKGANMVKDVLSNSTPLHSAARDGNEKLVKLLVKGADPTLVSLQNGCTPLHLAAKYNHLKIVKLLLQHGADVHKVDLSGLTPLHLAAKYGHFSVAQLLLEHGADVHARDLSGLTPLHLAAKFGHTDCVKLLLSHGADVNAKDSEGRTALHLAVDNRSDICKALLLKGAST",
        "PTGS2": "MLARALLLCAVLALSHTANPCCSHPCQNRGVCMSVGFDQYKCDCTRTGFYGENCTTPEFLTRIKLLLKPTPNTVHYILTHFKGVWNIVNNIPFLRSLIMKYVLTSRSYLIDSPPTYNVHYGYKSWEAFSNLSYYTRALPPVPDDCPTPLGVKGKKQLPDVNHFLLAQNSIKDVKEFSTKVQIPQKYQVVSDHLAEEEYQAKVDLYGRILAKDADNTIEMMHAKGNKKIIPATPQQGRGRLSVGSSILFSVTLCSVGICAVTYGFVNTGGLRWTVQPVPRTVGGHSGYHFGLAKPFSKTHEWAQHHLRYNFPVAVEPGVEVRLQDGTLMPDLYVATFYRLGLPGFWFQCHPVHKATQPIQLSIHSLPVLPDRKEVLNAVREIILDTLQFLRNHWIKDLKRGQAK",
        "IL10": "MHSSALLCCLVLLTGVRASPGQGTQSENSCTHFPGNLPNMLRDLRDAFSRVKTFFQMKDQLDNLLLKESLLEDFKGYLGCQALSEMIQFYLEEVMPQAENQDPDIKAHVNSLGENLKTLRLRLRRCHRFLPCENKSKAVEQVKNAFNKLQEKGIYKAMSEFDIFINYIEAYMTMKIRN",
        "CXCL8": "MTSKLAVALLAAFLISAALCEGAVLPRSAKELRCQCIKTYSKPFHPKFIKELRVIESGPHCANTEIIVKLSDGRELCLDPKENWVQRVVEKFLKRAENS",
        "CCL2": "MKVSAALLCLLLIAATFIPQGLAQPDAINAPVTCCYNFTNRKISVQRLASYRRITSSKCPKEAVIFKTIVAKEICADPKQKWVQDSMDHLDKQTQTPKT",
        "CASP3": "MENTENSVDSKSIKNLEPKIIHGSESMDSGISLDNSYKMDYPEMGLCIIINNKNFHKSTGMTSRSGTDVDAANLRETFRNLKYEVRNKNDLTREEIVELMRDVSKEDHSKRSSFVCVLLSHGEEGIIFGTNGPVDLKKITNFFRGDRCRSLTGKPKLFIIQACRGTELDCGIETDSGVDDDMACHKIPVEADFLYAYSTAPGYYSWRNSKDGSWFIQSLCAMLKQYADKLEFMHILTRVNRKVATEFESFSFDATFHAKKQIPCIVSMLTKELYFYH",
        "BCL2": "MAHAGRTGYDNREIVMKYIHYKLSQRGYEWDAGDVGAAPPGAAPAPGIFSSQPGHTPHPAASRDPVARTSPLQTPAAPGAAAGPALSPVPPVVHLTLRQAGDDFSRRYRRDFAEMSSQLHLTPFTARGRFATVVEELFRDGVNWGRIVAFFEFGGVMCVESVNREMSPLVDNIALWMTEYLNRHLHTWIQDNGGWDAFVELYGPSMRPLFDFSWLSLKTLLSLALVGACITLGAYLGHK",
        "BAX": "MDGSGEQPRGGGPTSSEQIMKTGALLLQGFIQDRAGRMGGEAPELALDPVPQDASTKKLSECLKRIGDELDSNMELQRMIAAVDTDSPREVFFRVAADMFSDGNFNWGRVVALFYFASKLVLKALCTKVPELIRTIMGWTLDFLRERLLGWIQDQGGWDGLLSYFGTPTWQTVTIFVAGVLTASLTIWKKMG",
        "TP53": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD",
        "AKT1": "MSDVAIVKEGWLHKRGEYIKTWRPRYFLLKNDGTFIGYKERPQDVDQREAPLNNFSVAQCQLMKTERPRPNTFIIRCLQWTTVIERTFHVETPEEREEWTTAIQTVADGLKKQEEEEMDFRSGSPSDNSGAEEMEVSLAKPKHRVTMNEFEYLKLLGKGTFGKVILVKEKATGRYYAMKILKKEVIVAKDEVAHTLTENRVLQNSRHPFLTALKYSFQTHDRLCFVMEYANGGELFFHLSRERVFSEDRARFYGAEIVSALDYLHSEKNVVYRDLKLENLMLDKDGHIKITDFGLCKEGIKDGATMKTFCGTPEYLAPEVLEDNDYGRAVDWWGLGVVMYEMMCGRLPFYNQDHEKLFELILMEEIRFPRTLGPEAKSLLSGLLKKDPKQRLGGGSEDAKEIMQHRFFAGIVWQHVYEKKLSPPFKPQVTSETDTRYFDEEFTAQMITITPPDQDDSMECVDSERRPHFPQFSYSASGTA",
        "MAPK1": "MAAAAAAGAGPEMVRGQVFDVGPRYTNLSYIGEGAYGMVCSAYDNVNKVRVAIKKISPFEHQTYCQRTLREIKILLRFRHENIIGINDIIRAPTIEQMKDVYIVQDLMETDLYKLLKTQHLSNDHICYFLYQILRGLKYIHSANVLHRDLKPSNLLLNTTCDLKICDFGLARVADPDHDHTGFLTEYVATRWYRAPEIMLNSKGYTKSIDIWSVGCILAEMLSNRPIFPGKHYLDQLNHILGILGSPSQEDLNCIINLKARNYLLSLPHKNKVPWNRLFPNADSKALDLLDKMLTFNPHKRIEVEQALAHPYLEQYYDPSDEPIAEAPFKFDMELDDLPKEKLKELIFEETARFQPGYRS"
    }
    
    # Amino acid frequencies for generating random sequences
    aa_frequencies = {
        'A': 0.074, 'C': 0.033, 'D': 0.059, 'E': 0.058, 'F': 0.040, 
        'G': 0.074, 'H': 0.029, 'I': 0.038, 'K': 0.072, 'L': 0.076, 
        'M': 0.018, 'N': 0.044, 'P': 0.050, 'Q': 0.037, 'R': 0.042, 
        'S': 0.081, 'T': 0.062, 'V': 0.068, 'W': 0.013, 'Y': 0.033
    }
    
    # Create sequence data
    sequences = []
    
    for target in targets:
        # Use real sequence if available, otherwise generate a synthetic one
        if target in target_sequences:
            sequence = target_sequences[target]
        else:
            # Generate random protein sequence with realistic amino acid frequencies
            length = random.randint(80, 200)  # Random length between 80-200 amino acids
            
            # Generate sequence by sampling from amino acid frequencies
            sequence = ''.join(random.choices(
                list(aa_frequencies.keys()), 
                weights=list(aa_frequencies.values()), 
                k=length
            ))
        
        # Add to sequences list
        sequences.append({
            'target_id': target,
            'sequence': sequence
        })
    
    return sequences

def calculate_drug_similarities(drug_structures_df, output_path):
    """Calculate and save drug structural similarities"""
    # Initialize calculator with Morgan fingerprints
    calculator = DrugSimilarityCalculator(fingerprint_type='morgan', radius=2, nBits=1024)
    
    # Calculate similarity matrix
    try:
        similarity_df, valid_drugs, fingerprints = calculator.calculate_similarity_matrix(
            drug_structures_df, smiles_col='smiles', id_col='compound_id'
        )
        
        # Save to CSV
        calculator.save_similarity_matrix(similarity_df, output_path)
        print(f"Drug similarity matrix calculated for {len(valid_drugs)} compounds and saved to {output_path}")
        
        return similarity_df
    except Exception as e:
        print(f"Error calculating drug similarities: {e}")
        return None

def calculate_protein_similarities(protein_sequences_df, output_path):
    """Calculate and save protein sequence similarities"""
    # For large datasets, use a faster, simpler similarity calculation
    # (full alignment would be too slow for many proteins)
    use_simple_method = len(protein_sequences_df) > 30
    
    if use_simple_method:
        # Use simplified calculator that approximates sequence similarity
        calculator = SimplifiedProteinSimilarityCalculator()
    else:
        # Use the full alignment-based calculator
        calculator = ProteinSimilarityCalculator(method='local', matrix='blosum62')
    
    # Calculate similarity matrix
    try:
        similarity_df, valid_proteins = calculator.calculate_similarity_matrix(
            protein_sequences_df, sequence_col='sequence', id_col='target_id'
        )
        
        # Save to CSV
        calculator.save_similarity_matrix(similarity_df, output_path)
        print(f"Protein similarity matrix calculated for {len(valid_proteins)} proteins and saved to {output_path}")
        
        return similarity_df
    except Exception as e:
        print(f"Error calculating protein similarities: {e}")
        return None

class SimplifiedProteinSimilarityCalculator:
    """
    Simplified calculator for protein sequence similarities using k-mer counting
    instead of full alignment (faster for large datasets)
    """
    def __init__(self, k=3):
        """
        Initialize protein similarity calculator with k-mer size
        
        Args:
            k: Size of k-mers to use for comparison
        """
        self.k = k
    
    def _get_kmer_counts(self, sequence):
        """Count k-mers in sequence"""
        kmer_counts = {}
        
        # Count k-mers
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i + self.k]
            kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
        
        return kmer_counts
    
    def calculate_sequence_similarity(self, seq1, seq2):
        """
        Calculate similarity between two protein sequences using k-mer counting
        
        Args:
            seq1: First protein sequence
            seq2: Second protein sequence
            
        Returns:
            Jaccard similarity between k-mer sets
        """
        # Handle edge cases
        if not seq1 or not seq2:
            return 0.0
            
        if seq1 == seq2:
            return 1.0
        
        # Get k-mer counts
        kmer_counts1 = self._get_kmer_counts(seq1)
        kmer_counts2 = self._get_kmer_counts(seq2)
        
        # Get all k-mers
        all_kmers = set(kmer_counts1.keys()) | set(kmer_counts2.keys())
        
        if not all_kmers:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = 0
        union = 0
        
        for kmer in all_kmers:
            count1 = kmer_counts1.get(kmer, 0)
            count2 = kmer_counts2.get(kmer, 0)
            
            intersection += min(count1, count2)
            union += max(count1, count2)
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate enhanced sample data for TCM target prioritization')
    parser.add_argument('--output', type=str, default='data', help='Output directory')
    parser.add_argument('--size', type=str, default='large', choices=['small', 'medium', 'large'],
                      help='Size of dataset to generate')
    
    args = parser.parse_args()
    
    # Generate enhanced sample data
    generate_enhanced_sample_data(args.output, args.size)
