#!/usr/bin/env python3
"""
Main script for TCM target prioritization system

This script provides the main functionality for training models and computing target priorities.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.data.graph_builder import build_semantic_graph
from src.models.fixed_graph_sage import FixedGraphSAGE
from src.models.rgcn_model import RGCN
from src.training.trainer import Trainer
from src.evaluation.metrics import calculate_metrics, plot_validation_metrics

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='TCM Target Prioritization')
    
    # Data arguments
    parser.add_argument('--kg_data', type=str, default='data/raw/kg_data.csv',
                        help='Knowledge graph data file')
    parser.add_argument('--db_data', type=str, default='data/raw/database_data.csv',
                        help='Database data file')
    parser.add_argument('--disease_importance', type=str, default='data/raw/disease_importance.csv',
                        help='Disease importance data file')
    parser.add_argument('--validated_data', type=str, default='data/raw/validated_interactions.csv',
                        help='Validated interactions data file')
    parser.add_argument('--drug_similarity', type=str, default=None,
                        help='Drug similarity matrix file')
    parser.add_argument('--protein_similarity', type=str, default=None,
                        help='Protein similarity matrix file')
    parser.add_argument('--semantic_similarities', type=str, default=None,
                        help='Semantic similarities file')
    parser.add_argument('--generate_samples', action='store_true',
                        help='Generate sample data instead of loading from files')
    
    # Model arguments
    parser.add_argument('--load_model', action='store_true',
                        help='Load pre-trained model')
    parser.add_argument('--model_path', type=str, default='results/models/rgcn_model.pt',
                        help='Path to pre-trained model')
    parser.add_argument('--save_model', action='store_true',
                        help='Save trained model')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='Margin for contrastive loss')
    parser.add_argument('--neg_samples', type=int, default=1,
                        help='Number of negative samples per positive sample')
    
    # Prioritization arguments
    parser.add_argument('--use_drug_similarity', action='store_true',
                        help='Use drug similarity for prioritization')
    parser.add_argument('--use_protein_similarity', action='store_true',
                        help='Use protein similarity for prioritization')
    parser.add_argument('--use_semantic_similarity', action='store_true',
                        help='Use semantic similarity for prioritization')
    parser.add_argument('--embedding_weight', type=float, default=0.05,
                        help='Weight for embedding similarity')
    parser.add_argument('--importance_weight', type=float, default=0.8,
                        help='Weight for disease importance')
    parser.add_argument('--drug_sim_weight', type=float, default=0.075,
                        help='Weight for drug similarity')
    parser.add_argument('--protein_sim_weight', type=float, default=0.075,
                        help='Weight for protein similarity')
    parser.add_argument('--semantic_weight', type=float, default=0.05,
                        help='Weight for semantic similarity')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of top targets to return')
    
    # Validation arguments
    parser.add_argument('--validate', action='store_true',
                        help='Validate model')
    
    return parser.parse_args()

def load_data(args):
    """
    Load data from files or generate sample data
    
    Args:
        args: Command-line arguments
        
    Returns:
        Tuple of (kg_data, db_data, disease_importance, validated_data, 
                 drug_similarity, protein_similarity, semantic_similarities)
    """
    # Check if generate_samples is set (with fallback to False)
    generate_samples = getattr(args, 'generate_samples', False)
    
    if generate_samples:
        # Generate sample data (not implemented in this version)
        print("Generating sample data not implemented in this version.")
        return None, None, None, None, None, None, None
    
    # Load knowledge graph data
    kg_data = pd.read_csv(args.kg_data)
    print(f"Loaded knowledge graph with {len(kg_data)} relations")
    
    # Load database data
    db_data = pd.read_csv(args.db_data)
    print(f"Loaded database with {len(db_data)} compounds")
    
    # Load disease importance data
    disease_importance = pd.read_csv(args.disease_importance)
    print(f"Loaded disease importance data with {len(disease_importance)} entries")
    
    # Load validated interactions if provided
    validated_data = None
    if args.validated_data and os.path.exists(args.validated_data):
        validated_data = pd.read_csv(args.validated_data)
        print(f"Loaded {len(validated_data)} validated interactions")
    
    # Load drug similarity matrix if provided
    drug_similarity = None
    if args.drug_similarity and os.path.exists(args.drug_similarity):
        print(f"Loading drug similarity matrix from {args.drug_similarity}")
        drug_similarity = pd.read_csv(args.drug_similarity, index_col=0)
    
    # Load protein similarity matrix if provided
    protein_similarity = None
    if args.protein_similarity and os.path.exists(args.protein_similarity):
        print(f"Loading protein similarity matrix from {args.protein_similarity}")
        protein_similarity = pd.read_csv(args.protein_similarity, index_col=0)
    
    # Load semantic similarities if provided
    semantic_similarities = None
    if args.semantic_similarities and os.path.exists(args.semantic_similarities):
        print(f"Loading semantic similarities from {args.semantic_similarities}")
        semantic_similarities = pd.read_csv(args.semantic_similarities)
    
    return kg_data, db_data, disease_importance, validated_data, drug_similarity, protein_similarity, semantic_similarities

def train_or_load_model(args, data, important_targets):
    """
    Train a new model or load a pre-trained model
    
    Args:
        args: Command-line arguments
        data: PyG Data object
        important_targets: List of important target indices
        
    Returns:
        Tuple of (embeddings, model)
    """
    # Get device - handle case where args.cpu might not exist (for backward compatibility)
    use_cpu = getattr(args, 'cpu', False)
    device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    in_dim = data.x.shape[1]
    hidden_dim = 256
    out_dim = 128
    
    # Use RGCN model if semantic relations are used
    if args.use_semantic_similarity:
        from src.models.rgcn_model import RGCN
        model = RGCN(in_dim, hidden_dim, out_dim, 
                    num_relations=data.num_relations, 
                    dropout=0.3).to(device)
    else:
        from src.models.fixed_graph_sage import FixedGraphSAGE
        model = FixedGraphSAGE(in_dim, hidden_dim, out_dim, dropout=0.3).to(device)
    
    # Train or load model
    if args.load_model and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        
        # Generate embeddings
        with torch.no_grad():
            data = data.to(device)
            # Always pass edge_type regardless of model type for compatibility
            embeddings = model(data.x, data.edge_index, data.edge_type, data.edge_weight)
    else:
        print("Training model...")
        # Initialize trainer
        from src.training.trainer import Trainer
        trainer = Trainer(
            model=model, 
            device=device,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            margin=args.margin,
            neg_samples=args.neg_samples,
            save_model=args.save_model
        )
        
        print(f"Trainer initialized with device: {device}")
        print("Training model...")
        
        # Train model and get embeddings
        embeddings = trainer.train(data, important_targets)
    
    return embeddings, model

# Find target column in disease importance dataframe (called within calculate_target_priorities_with_semantics)
def find_target_column(dataframe):
    """
    Find the target identifier column in a dataframe
    
    Args:
        dataframe: DataFrame to search
        
    Returns:
        Column name for target identifier
    """
    target_id_columns = ['target_id', 'target', 'protein_id', 'protein', 'gene_id', 'gene']
    
    for col in target_id_columns:
        if col in dataframe.columns:
            return col
    
    # If no known column is found, use the first column as a fallback
    if len(dataframe.columns) > 0:
        print(f"WARNING: Could not find target identifier column. Using '{dataframe.columns[0]}'.")
        return dataframe.columns[0]
    
    return None


# Modified section from calculate_target_priorities_with_semantics function
# This would replace the corresponding section in the full function
def get_importance_scores(target_indices, reverse_node_map, disease_importance):
    """
    Get importance scores for targets from disease importance data
    
    Args:
        target_indices: Target indices
        reverse_node_map: Mapping from indices to node IDs
        disease_importance: DataFrame with disease importance scores
        
    Returns:
        Dictionary of target importance scores
    """
    # Find target column in disease importance dataframe
    target_col = find_target_column(disease_importance)
    
    if target_col is None:
        print("WARNING: No columns found in disease importance dataframe. Using empty importance scores.")
        return {}
    
    # Find importance column
    importance_col = None
    importance_columns = ['importance_score', 'importance', 'score', 'weight', 'priority']
    
    for col in importance_columns:
        if col in disease_importance.columns:
            importance_col = col
            break
    
    if importance_col is None and len(disease_importance.columns) > 1:
        # If no known column is found, use the second column
        importance_col = disease_importance.columns[1]
        print(f"WARNING: Could not find importance score column. Using '{importance_col}'.")
    elif importance_col is None:
        print("WARNING: No importance score column found. Using default importance of 1.0.")
        
    # Calculate importance scores for all targets
    importance_scores = {}
    
    for target_idx in target_indices:
        target_id = reverse_node_map[target_idx]
        
        # Get importance score from data
        if target_col is not None and importance_col is not None:
            mask = disease_importance[target_col] == target_id
            if mask.any():
                importance = disease_importance.loc[mask, importance_col].values[0]
                importance_scores[target_id] = float(importance)
            else:
                importance_scores[target_id] = 0.0
        else:
            # If no valid columns, use default importance
            importance_scores[target_id] = 1.0
    
    return importance_scores

def calculate_drug_similarity_scores(target_indices, reverse_node_map, compound_id, node_map, data, drug_similarity):
    """
    Calculate drug-based similarity scores for all targets
    
    Args:
        target_indices: Target indices
        reverse_node_map: Mapping from indices to node IDs
        compound_id: Compound ID to prioritize targets for
        node_map: Mapping from node IDs to indices
        data: PyG Data object
        drug_similarity: DataFrame with drug similarity matrix
        
    Returns:
        Dictionary of drug similarity scores for targets
    """
    drug_sim_scores = {}
    compound_idx = node_map[compound_id]
    
    # Check if compound exists in drug similarity matrix
    if compound_id not in drug_similarity.index:
        print(f"WARNING: Compound '{compound_id}' not found in drug similarity matrix. Using default scores of 0.0.")
        # Return zeros for all targets
        for target_idx in target_indices:
            target_id = reverse_node_map[target_idx]
            drug_sim_scores[target_id] = 0.0
        return drug_sim_scores
    
    # Get all targets connected to similar compounds
    for target_idx in target_indices:
        target_id = reverse_node_map[target_idx]
        
        # Get all compounds similar to the query compound
        similar_compounds = []
        for other_compound in drug_similarity.columns:
            # Skip compounds not in the index or the compound itself
            if other_compound != compound_id and other_compound in drug_similarity.index:
                try:
                    sim = drug_similarity.loc[compound_id, other_compound]
                    if sim > 0:  # Only consider positive similarities
                        similar_compounds.append((other_compound, sim))
                except KeyError:
                    # Skip if there's any issue with accessing similarity
                    continue
        
        # Check if target is connected to any similar compound
        score = 0.0
        
        for other_compound, sim in similar_compounds:
            if other_compound in node_map:
                other_idx = node_map[other_compound]
                
                # Check if there's an edge between similar compound and target
                edge_indices = data.edge_index.cpu().numpy()
                for i in range(edge_indices.shape[1]):
                    if (edge_indices[0, i] == other_idx and edge_indices[1, i] == target_idx) or \
                       (edge_indices[0, i] == target_idx and edge_indices[1, i] == other_idx):
                        # Weight by similarity between compounds
                        score += sim
                        break
        
        drug_sim_scores[target_id] = score
    
    # Normalize drug similarity scores
    max_drug_sim = max(drug_sim_scores.values()) if drug_sim_scores else 1.0
    if max_drug_sim > 0:
        for target_id in drug_sim_scores:
            drug_sim_scores[target_id] /= max_drug_sim
    
    return drug_sim_scores

def calculate_target_priorities_with_semantics(embeddings, data, compound_id, node_map, reverse_node_map, 
                                 disease_importance, drug_similarity=None, protein_similarity=None,
                                 semantic_similarities=None,
                                 embedding_weight=0.05, importance_weight=0.8,
                                 drug_sim_weight=0.05, protein_sim_weight=0.05,
                                 semantic_weight=0.05, k=10, save_results=True):
    """
    Calculate target priorities using multiple components including semantic similarities
    
    Args:
        embeddings: Node embeddings from GNN
        data: PyG Data object
        compound_id: Compound ID to prioritize targets for
        node_map: Mapping from node IDs to indices
        reverse_node_map: Mapping from indices to node IDs
        disease_importance: DataFrame with disease importance scores
        drug_similarity: DataFrame with drug similarity matrix
        protein_similarity: DataFrame with protein similarity matrix
        semantic_similarities: DataFrame with semantic similarities
        embedding_weight: Weight for embedding similarity
        importance_weight: Weight for disease importance
        drug_sim_weight: Weight for drug similarity
        protein_sim_weight: Weight for protein similarity
        semantic_weight: Weight for semantic similarity
        k: Number of top targets to return
        save_results: Whether to save results
        
    Returns:
        DataFrame with target priorities
    """
    # Normalize weights to sum to 1.0
    total_weight = embedding_weight + importance_weight + drug_sim_weight + protein_sim_weight + semantic_weight
    embedding_weight /= total_weight
    importance_weight /= total_weight
    drug_sim_weight /= total_weight
    protein_sim_weight /= total_weight
    semantic_weight /= total_weight
    
    print(f"Normalized weights: Embedding={embedding_weight:.4f}, Importance={importance_weight:.4f}, " +
          f"Drug Sim={drug_sim_weight:.4f}, Protein Sim={protein_sim_weight:.4f}, Semantic={semantic_weight:.4f}")
    
    # Extract target indices
    target_indices = data.target_indices.cpu().numpy()
    
    # Get compound index
    if compound_id not in node_map:
        raise ValueError(f"Compound {compound_id} not found in graph")
    compound_idx = node_map[compound_id]
    
    # Extract compound and target embeddings
    compound_embedding = embeddings[compound_idx].cpu().numpy()
    
    # Calculate embedding similarity for all targets
    similarities = {}
    
    for target_idx in target_indices:
        target_id = reverse_node_map[target_idx]
        target_embedding = embeddings[target_idx].cpu().numpy()
        
        # Calculate cosine similarity
        sim = np.dot(compound_embedding, target_embedding) / \
              (np.linalg.norm(compound_embedding) * np.linalg.norm(target_embedding))
        
        similarities[target_id] = sim
    
    # Calculate disease importance scores for all targets using our helper function
    importance_scores = get_importance_scores(target_indices, reverse_node_map, disease_importance)
    
    # Apply non-linear transformation to enhance contrast in importance scores
    max_importance = max(importance_scores.values()) if importance_scores else 1.0
    for target_id in importance_scores:
        # Normalize importance
        norm_importance = importance_scores[target_id] / max_importance
        # Apply power transformation
        adjusted_importance = np.power(norm_importance, 1.5)  # Power transformation
        importance_scores[target_id] = adjusted_importance
    
    # Calculate drug-based similarity for all targets using our helper function
    drug_sim_scores = {}
    if drug_similarity is not None and drug_sim_weight > 0:
        drug_sim_scores = calculate_drug_similarity_scores(
            target_indices, reverse_node_map, compound_id, node_map, data, drug_similarity
        )
    
    # Calculate protein-based similarity for all targets
    protein_sim_scores = {}
    
    if protein_similarity is not None and protein_sim_weight > 0:
        # First, find targets known to interact with the compound
        known_targets = []
        edge_indices = data.edge_index.cpu().numpy()
        
        for i in range(edge_indices.shape[1]):
            if edge_indices[0, i] == compound_idx:
                target_idx = edge_indices[1, i]
                if target_idx in target_indices:
                    known_targets.append(reverse_node_map[target_idx])
            elif edge_indices[1, i] == compound_idx:
                target_idx = edge_indices[0, i]
                if target_idx in target_indices:
                    known_targets.append(reverse_node_map[target_idx])
        
        # Calculate protein similarity-based scores
        for target_idx in target_indices:
            target_id = reverse_node_map[target_idx]
            
            # Skip if target is already known to interact with compound
            if target_id in known_targets:
                protein_sim_scores[target_id] = 1.0
                continue
            
            # Calculate protein similarity to known targets
            max_sim = 0.0
            
            for known_target in known_targets:
                # Handle case where target IDs might not be in similarity matrix
                if known_target in protein_similarity.index and target_id in protein_similarity.columns:
                    try:
                        sim = protein_similarity.loc[known_target, target_id]
                        max_sim = max(max_sim, sim)
                    except KeyError:
                        continue
            
            protein_sim_scores[target_id] = max_sim
    
    # Calculate semantic similarity scores for all targets
    semantic_sim_scores = {}
    
    if semantic_similarities is not None and semantic_weight > 0:
        # Find compound column in semantic similarities
        compound_col = None
        compound_columns = ['compound_id', 'compound', 'drug_id', 'drug']
        
        for col in compound_columns:
            if col in semantic_similarities.columns:
                compound_col = col
                break
        
        if compound_col is None and len(semantic_similarities.columns) > 0:
            compound_col = semantic_similarities.columns[0]
            print(f"WARNING: Could not find compound column in semantic similarities. Using '{compound_col}'.")
        
        # Find target column in semantic similarities
        target_col = None
        target_columns = ['target_id', 'target', 'protein_id', 'protein']
        
        for col in target_columns:
            if col in semantic_similarities.columns:
                target_col = col
                break
        
        if target_col is None and len(semantic_similarities.columns) > 1:
            target_col = semantic_similarities.columns[1]
            print(f"WARNING: Could not find target column in semantic similarities. Using '{target_col}'.")
        
        # Find similarity column in semantic similarities
        sim_col = None
        sim_columns = ['semantic_similarity', 'similarity', 'sim_score', 'score']
        
        for col in sim_columns:
            if col in semantic_similarities.columns:
                sim_col = col
                break
        
        if sim_col is None and len(semantic_similarities.columns) > 2:
            sim_col = semantic_similarities.columns[2]
            print(f"WARNING: Could not find similarity column in semantic similarities. Using '{sim_col}'.")
        
        # Calculate semantic similarity scores
        if compound_col is not None and target_col is not None and sim_col is not None:
            for target_idx in target_indices:
                target_id = reverse_node_map[target_idx]
                
                # Check if semantic similarity exists for this compound-target pair
                mask = (semantic_similarities[compound_col] == compound_id) & \
                      (semantic_similarities[target_col] == target_id)
                
                if mask.any():
                    semantic_sim_scores[target_id] = semantic_similarities.loc[mask, sim_col].values[0]
                else:
                    semantic_sim_scores[target_id] = 0.0
    
    # Combine scores
    final_scores = {}
    
    for target_idx in target_indices:
        target_id = reverse_node_map[target_idx]
        
        # Get scores from each component (default to 0 if not available)
        emb_sim = similarities.get(target_id, 0.0)
        imp_score = importance_scores.get(target_id, 0.0)
        drug_score = drug_sim_scores.get(target_id, 0.0) if drug_sim_weight > 0 else 0.0
        protein_score = protein_sim_scores.get(target_id, 0.0) if protein_sim_weight > 0 else 0.0
        semantic_score = semantic_sim_scores.get(target_id, 0.0) if semantic_weight > 0 else 0.0
        
        # Weighted combination
        final_score = (embedding_weight * emb_sim) + \
                      (importance_weight * imp_score) + \
                      (drug_sim_weight * drug_score) + \
                      (protein_sim_weight * protein_score) + \
                      (semantic_weight * semantic_score)
        
        final_scores[target_id] = final_score
    
    # Sort targets by final score
    sorted_targets = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Create result DataFrame
    results = []
    
    for target_id, final_score in sorted_targets[:k]:
        # Get individual scores
        emb_sim = similarities.get(target_id, 0.0)
        imp_score = importance_scores.get(target_id, 0.0)
        drug_score = drug_sim_scores.get(target_id, 0.0) if drug_sim_weight > 0 else 0.0
        protein_score = protein_sim_scores.get(target_id, 0.0) if protein_sim_weight > 0 else 0.0
        semantic_score = semantic_sim_scores.get(target_id, 0.0) if semantic_weight > 0 else 0.0
        
        results.append({
            'compound_id': compound_id,
            'target_id': target_id,
            'final_score': final_score,
            'embedding_similarity': emb_sim,
            'importance_score': imp_score,
            'drug_similarity_score': drug_score,
            'protein_similarity_score': protein_score,
            'semantic_similarity_score': semantic_score
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    if save_results:
        os.makedirs('results/prioritization', exist_ok=True)
        df.to_csv(f'results/prioritization/target_priorities_{compound_id}.csv', index=False)
        print(f"Results saved to results/prioritization/target_priorities_{compound_id}.csv")
    
    return df

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

def extract_important_targets(disease_importance, node_map):
    """
    Extract important targets for training
    
    Args:
        disease_importance: DataFrame with disease importance scores
        node_map: Mapping from node IDs to indices
        
    Returns:
        List of important target indices
    """
    important_targets = []
    
    # Check column names to find the target identifier column
    target_id_columns = ['target_id', 'target', 'protein_id', 'protein', 'gene_id', 'gene']
    target_col = None
    
    for col in target_id_columns:
        if col in disease_importance.columns:
            target_col = col
            break
    
    if target_col is None:
        print("WARNING: Could not find target identifier column in disease_importance dataframe.")
        print(f"Available columns: {list(disease_importance.columns)}")
        print("Using first column as target identifier.")
        target_col = disease_importance.columns[0]
    
    print(f"Using '{target_col}' as target identifier column.")
    
    # Extract target indices
    for _, row in disease_importance.iterrows():
        target_id = row[target_col]
        
        if target_id in node_map:
            target_idx = node_map[target_id]
            important_targets.append(target_idx)
    
    return important_targets

def create_visualization(embeddings, data, node_map, reverse_node_map, predictions):
    """
    Create visualizations of embeddings and predictions
    
    Args:
        embeddings: Node embeddings
        data: PyG Data object
        node_map: Mapping from node IDs to indices
        reverse_node_map: Mapping from indices to node IDs
        predictions: DataFrame with target predictions
    """
    # Create output directory
    os.makedirs('results/visualizations', exist_ok=True)
    
    # Convert embeddings to numpy
    embeddings_np = embeddings.detach().cpu().numpy()
    
    # Extract compound and target indices
    compound_indices = data.compound_indices.cpu().numpy()
    target_indices = data.target_indices.cpu().numpy()
    
    # Get embedding labels
    labels = []
    node_ids = []
    
    for i in range(embeddings_np.shape[0]):
        node_id = reverse_node_map[i]
        node_ids.append(node_id)
        
        if i in compound_indices:
            labels.append('Compound')
        elif i in target_indices:
            labels.append('Target')
        else:
            labels.append('Unknown')
    
    # Apply t-SNE with reduced perplexity appropriate for small datasets
    # The perplexity should be smaller than the number of samples
    n_samples = embeddings_np.shape[0]
    perplexity = min(n_samples - 1, 15)  # Use 15 or fewer if we have a small graph
    
    try:
        print(f"Applying t-SNE with perplexity={perplexity} for {n_samples} nodes...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings_np)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'label': labels,
            'node_id': node_ids
        })
        
        # Plot embeddings
        plt.figure(figsize=(10, 8))
        
        # Plot compounds
        compounds = df[df['label'] == 'Compound']
        plt.scatter(compounds['x'], compounds['y'], c='blue', marker='o', s=100, alpha=0.7, label='Compound')
        
        # Plot targets
        targets = df[df['label'] == 'Target']
        plt.scatter(targets['x'], targets['y'], c='red', marker='^', s=100, alpha=0.7, label='Target')
        
        # Add node IDs as labels
        for _, row in df.iterrows():
            plt.annotate(row['node_id'], (row['x'], row['y']), fontsize=8, alpha=0.8)
        
        plt.title('t-SNE Visualization of Node Embeddings')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        plt.savefig('results/visualizations/embeddings_tsne.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Embeddings visualization saved to results/visualizations/embeddings_tsne.png")
    except Exception as e:
        print(f"Warning: t-SNE visualization failed: {str(e)}")
        print("Skipping t-SNE visualization and continuing...")
    
    # Create heatmap of top predictions
    try:
        top_compounds = predictions['compound_id'].unique()[:5]  # Take top 5 compounds
        
        if len(top_compounds) == 0:
            print("No compounds in predictions, skipping heatmap")
            return
        
        # Filter predictions for top compounds
        filtered_predictions = predictions[predictions['compound_id'].isin(top_compounds)]
        
        # Get top targets across all filtered compounds
        top_targets = filtered_predictions['target_id'].value_counts().head(10).index.tolist()
        
        if len(top_targets) == 0:
            print("No targets in filtered predictions, skipping heatmap")
            return
        
        # Create matrix for heatmap
        heatmap_data = np.zeros((len(top_compounds), len(top_targets)))
        
        for i, compound_id in enumerate(top_compounds):
            for j, target_id in enumerate(top_targets):
                # Find score for this compound-target pair
                mask = (filtered_predictions['compound_id'] == compound_id) & \
                      (filtered_predictions['target_id'] == target_id)
                
                if mask.any():
                    heatmap_data[i, j] = filtered_predictions.loc[mask, 'final_score'].values[0]
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu",
                  xticklabels=top_targets, yticklabels=top_compounds)
        plt.title('Top Compound-Target Prioritization Scores')
        plt.xlabel('Target IDs')
        plt.ylabel('Compound IDs')
        
        # Save heatmap
        plt.savefig('results/visualizations/priority_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Priority heatmap saved to results/visualizations/priority_heatmap.png")
    except Exception as e:
        print(f"Warning: Heatmap visualization failed: {str(e)}")
        print("Skipping heatmap visualization and continuing...")

def validate_predictions(predictions, validated_data):
    """
    Validate predictions against validated interactions
    
    Args:
        predictions: DataFrame with target predictions
        validated_data: DataFrame with validated interactions
    """
    try:
        # Ensure output directory exists
        os.makedirs('results/evaluation', exist_ok=True)
        
        # Standardize column names in validated data
        if 'compound_id' not in validated_data.columns and 'compound' in validated_data.columns:
            validated_data = validated_data.rename(columns={'compound': 'compound_id'})
        
        if 'target_id' not in validated_data.columns and 'target' in validated_data.columns:
            validated_data = validated_data.rename(columns={'target': 'target_id'})
        
        # Check if necessary columns exist
        required_columns = ['compound_id', 'target_id']
        for col in required_columns:
            if col not in validated_data.columns:
                print(f"Warning: '{col}' column not found in validated data. Available columns: {validated_data.columns.tolist()}")
                print("Skipping validation due to missing columns.")
                return
            
            if col not in predictions.columns:
                print(f"Warning: '{col}' column not found in predictions. Available columns: {predictions.columns.tolist()}")
                print("Skipping validation due to missing columns.")
                return
        
        # Calculate evaluation metrics
        metrics = calculate_metrics(predictions, validated_data)
        
        # Print metrics
        print("\n=== Validation Metrics ===")
        print(f"Precision-Recall AUC: {metrics['pr_auc']:.4f}" if not np.isnan(metrics['pr_auc']) else "Precision-Recall AUC: N/A")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}" if not np.isnan(metrics['roc_auc']) else "ROC AUC: N/A")
        print(f"Mean Reciprocal Rank: {metrics['mrr']:.4f}")
        
        for k in sorted([int(k.split('@')[1]) for k in metrics.keys() if k.startswith('hit@')]):
            print(f"Hit@{k}: {metrics[f'hit@{k}']:.4f}")
        
        # Plot metrics
        try:
            fig = plot_validation_metrics(metrics, save_path='results/evaluation/validation_results.png')
            plt.close(fig)
            print("Validation metrics plot saved to results/evaluation/validation_results.png")
        except Exception as e:
            print(f"Warning: Failed to create validation metrics plot: {str(e)}")
        
        # Save metrics to CSV
        metrics_to_save = {
            'Metric': ['PR AUC', 'ROC AUC', 'MRR'] + [f'Hit@{k}' for k in sorted([int(k.split('@')[1]) for k in metrics.keys() if k.startswith('hit@')])],
            'Value': [
                metrics['pr_auc'] if not np.isnan(metrics['pr_auc']) else "N/A",
                metrics['roc_auc'] if not np.isnan(metrics['roc_auc']) else "N/A",
                metrics['mrr']
            ] + [metrics[f'hit@{k}'] for k in sorted([int(k.split('@')[1]) for k in metrics.keys() if k.startswith('hit@')])]
        }
        pd.DataFrame(metrics_to_save).to_csv('results/evaluation/metrics.csv', index=False)
        print("Validation metrics saved to results/evaluation/metrics.csv")
    
    except Exception as e:
        print(f"Warning: Validation failed with error: {str(e)}")
        print("Continuing without validation...")

def main():
    """Main function"""
    # Parse command-line arguments
    args = parse_args()
    
    # Load data
    print("Loading data...")
    kg_data, db_data, disease_importance, validated_data, drug_similarity, protein_similarity, semantic_similarities = load_data(args)
    
    # Build graph
    print("Building graph with semantic relationships...")
    data, node_map, reverse_node_map, relation_types = build_semantic_graph(
        kg_data=kg_data,
        drug_similarity=drug_similarity if args.use_drug_similarity else None,
        protein_similarity=protein_similarity if args.use_protein_similarity else None
    )
    
    # Extract important targets
    important_targets = extract_important_targets(disease_importance, node_map)
    print(f"Found {len(important_targets)} important targets for training")
    
    # Train or load model
    embeddings, model = train_or_load_model(args, data, important_targets)
    
    # Calculate target priorities
    results = []
    
    for _, row in db_data.iterrows():
        compound_id = row['compound_id']
        
        # Skip if compound not in graph
        if compound_id not in node_map:
            print(f"Warning: Compound {compound_id} not found in graph. Skipping...")
            continue
        
        # Calculate target priorities
        df = calculate_target_priorities_with_semantics(
            embeddings=embeddings,
            data=data,
            compound_id=compound_id,
            node_map=node_map,
            reverse_node_map=reverse_node_map,
            disease_importance=disease_importance,
            drug_similarity=drug_similarity if args.use_drug_similarity else None,
            protein_similarity=protein_similarity if args.use_protein_similarity else None,
            semantic_similarities=semantic_similarities if args.use_semantic_similarity else None,
            embedding_weight=args.embedding_weight,
            importance_weight=args.importance_weight,
            drug_sim_weight=args.drug_sim_weight if args.use_drug_similarity else 0.0,
            protein_sim_weight=args.protein_sim_weight if args.use_protein_similarity else 0.0,
            semantic_weight=args.semantic_weight if args.use_semantic_similarity else 0.0,
            k=args.top_k
        )
        
        results.append(df)
    
    # Combine results
    if results:
        all_results = pd.concat(results, ignore_index=True)
        
        # Save combined results
        os.makedirs('results/prioritization', exist_ok=True)
        all_results.to_csv('results/prioritization/target_priorities.csv', index=False)
        print(f"Combined results saved to results/prioritization/target_priorities.csv")
        
        # Create visualizations
        create_visualization(embeddings, data, node_map, reverse_node_map, all_results)
        
        # Validate if specified
        if args.validate and validated_data is not None:
            validate_predictions(all_results, validated_data)
    else:
        print("Warning: No predictions generated")
    
    print("Done!")

if __name__ == "__main__":
    main()
