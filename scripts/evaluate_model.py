#!/usr/bin/env python
"""
Enhanced evaluation script for TCM target prioritization

This script implements advanced evaluation methods for assessing the performance
of the target prioritization system, including cross-validation, clustering analysis,
and stability assessment.
"""
import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.graph_builder import build_semantic_graph
from src.models.rgcn_model import RGCN
from src.training.trainer import Trainer
from src.evaluation.validation import validate_model
from src.evaluation.cross_validation import leave_one_out_validation, k_fold_cross_validation
from src.evaluation.clustering import analyze_embedding_space
from src.evaluation.overlap_analysis import calculate_top_n_overlap, analyze_stability

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced TCM Target Prioritization Evaluation')
    
    # Input data options
    parser.add_argument('--kg_data', type=str, default='data/raw/kg_data_extended.csv',
                      help='Knowledge graph data file')
    parser.add_argument('--validated_data', type=str, default='data/raw/validated_interactions.csv',
                      help='Validated interactions file')
    parser.add_argument('--disease_importance', type=str, default='data/raw/disease_importance_extended.csv',
                      help='Disease importance data file')
    
    # Model options
    parser.add_argument('--model_path', type=str, default='results/models/rgcn_model.pt',
                      help='Trained model file')
    parser.add_argument('--embeddings_path', type=str, default='results/embeddings/node_embeddings.pt',
                      help='Pre-computed embeddings file')
    
    # Evaluation options
    parser.add_argument('--loocv', action='store_true',
                      help='Perform leave-one-out cross-validation')
    parser.add_argument('--kfold', type=int, default=0,
                      help='Perform k-fold cross-validation with specified k')
    parser.add_argument('--clustering', action='store_true',
                      help='Perform embedding space clustering analysis')
    parser.add_argument('--stability', type=int, default=0,
                      help='Perform stability analysis with specified number of runs')
    parser.add_argument('--overlap', type=int, default=0,
                      help='Perform target overlap analysis with specified number of runs')
    
    # Weight parameters for prioritization
    parser.add_argument('--embedding_weight', type=float, default=0.05,
                      help='Weight for embedding similarity (default: 0.05)')
    parser.add_argument('--importance_weight', type=float, default=0.8,
                      help='Weight for target importance (default: 0.8)')
    parser.add_argument('--drug_sim_weight', type=float, default=0.075,
                      help='Weight for drug similarity (default: 0.075)')
    parser.add_argument('--protein_sim_weight', type=float, default=0.075,
                      help='Weight for protein similarity (default: 0.075)')
    parser.add_argument('--semantic_weight', type=float, default=0.0,
                      help='Weight for semantic similarity (default: 0.0)')
    
    # Other options
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                      help='Output directory')
    
    return parser.parse_args()

def load_data(args):
    """Load input data"""
    # Load knowledge graph data
    kg_data = pd.read_csv(args.kg_data)
    print(f"Loaded knowledge graph with {len(kg_data)} relations")
    
    # Load validated interactions
    validated_data = pd.read_csv(args.validated_data)
    print(f"Loaded {len(validated_data)} validated interactions")
    
    # Load disease importance data
    disease_data = pd.read_csv(args.disease_importance)
    print(f"Loaded disease importance data with {len(disease_data)} entries")
    
    # Extract target importance scores
    target_importance = {}
    for _, row in disease_data.iterrows():
        target = row.get('target_id', row.get('target', ''))
        importance = row['importance_score']
        
        # Take maximum importance across diseases
        if target not in target_importance or importance > target_importance[target]:
            target_importance[target] = float(importance)
    
    return kg_data, validated_data, target_importance

def load_or_train_model(args, data):
    """Load pre-trained model or train new one"""
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = RGCN(
        in_dim=data.x.shape[1],
        hidden_dim=256,
        out_dim=128,
        num_relations=data.num_relations,
        num_bases=4,
        dropout=0.3
    ).to(device)
    
    # Load model if path exists
    if os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print("Model file not found. Training new model...")
        
        # Create trainer
        trainer = Trainer(
            model=model,
            epochs=100,
            lr=0.0001,
            weight_decay=1e-5,
            margin=0.4,
            patience=20,
            device=device
        )
        
        # Move data to device
        data = data.to(device)
        
        # Train model
        embeddings = trainer.train(data)
        
        # Save model
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        torch.save(model.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")
        
        # Save embeddings
        os.makedirs(os.path.dirname(args.embeddings_path), exist_ok=True)
        torch.save(embeddings, args.embeddings_path)
        print(f"Embeddings saved to {args.embeddings_path}")
    
    # Get embeddings
    if os.path.exists(args.embeddings_path):
        print(f"Loading embeddings from {args.embeddings_path}")
        embeddings = torch.load(args.embeddings_path, map_location=device)
    else:
        print("Computing embeddings...")
        model.eval()
        with torch.no_grad():
            data = data.to(device)
            embeddings = model(
                data.x, 
                data.edge_index, 
                data.edge_type, 
                edge_weight=data.edge_weight if hasattr(data, 'edge_weight') else None
            )
    
    return model, embeddings

def calculate_target_priorities(embeddings, node_map, reverse_node_map, data,
                               target_importance, validated_data,
                               semantic_similarities=None, top_k=20, 
                               weights=None):
    """
    Enhanced target prioritization function with semantic similarity support
    
    Args:
        embeddings: Node embeddings
        node_map: Mapping from node IDs to indices
        reverse_node_map: Mapping from indices to node IDs
        data: Graph data object
        target_importance: Dictionary of target importance scores
        validated_data: DataFrame with validated interactions (for drug/protein similarity)
        semantic_similarities: DataFrame with semantic similarities (optional)
        top_k: Number of top targets to return
        weights: Dictionary of component weights
        
    Returns:
        Dictionary mapping compounds to prioritized targets
    """
    print("Calculating target priorities...")
    
    # Set default weights if not provided
    if weights is None:
        weights = {
            'embedding': 0.05,
            'importance': 0.8,
            'drug_sim': 0.075,
            'protein_sim': 0.075,
            'semantic': 0.0
        }
    
    # Ensure weights sum to 1.0
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-6:
        weights = {k: v / total_weight for k, v in weights.items()}
    
    # Extract compound and target indices
    compound_indices = data.compound_indices
    target_indices = data.target_indices
    
    # Convert to Python lists for easier indexing
    compound_list = [reverse_node_map[idx.item()] for idx in compound_indices]
    target_list = [reverse_node_map[idx.item()] for idx in target_indices]
    
    # Create semantic similarity matrix if provided
    semantic_matrix = None
    if semantic_similarities is not None and weights['semantic'] > 0:
        semantic_matrix = np.zeros((len(compound_list), len(target_list)))
        
        # Create maps from IDs to indices
        compound_to_idx = {comp_id: idx for idx, comp_id in enumerate(compound_list)}
        target_to_idx = {target_id: idx for idx, target_id in enumerate(target_list)}
        
        # Fill matrix
        for _, row in semantic_similarities.iterrows():
            compound_id = row['compound_id']
            target_id = row['target_id']
            
            if compound_id in compound_to_idx and target_id in target_to_idx:
                comp_idx = compound_to_idx[compound_id]
                target_idx = target_to_idx[target_id]
                semantic_matrix[comp_idx, target_idx] = row['semantic_similarity']
    
    # Create drug and protein similarity matrices from validated data
    drug_sim_matrix = np.zeros((len(compound_list), len(compound_list)))
    protein_sim_matrix = np.zeros((len(target_list), len(target_list)))
    
    # Calculate drug similarities based on shared targets
    for i, comp_i in enumerate(compound_list):
        targets_i = set()
        for _, row in validated_data.iterrows():
            if row.get('compound', row.get('compound_id', '')) == comp_i:
                target = row.get('target', row.get('target_id', ''))
                if target in target_list:
                    targets_i.add(target)
        
        for j, comp_j in enumerate(compound_list):
            if i != j:
                targets_j = set()
                for _, row in validated_data.iterrows():
                    if row.get('compound', row.get('compound_id', '')) == comp_j:
                        target = row.get('target', row.get('target_id', ''))
                        if target in target_list:
                            targets_j.add(target)
                
                # Calculate Jaccard similarity
                intersection = targets_i & targets_j
                union = targets_i | targets_j
                
                if union:
                    drug_sim_matrix[i, j] = len(intersection) / len(union)
    
    # Calculate protein similarities based on shared compounds
    for i, target_i in enumerate(target_list):
        compounds_i = set()
        for _, row in validated_data.iterrows():
            if row.get('target', row.get('target_id', '')) == target_i:
                compound = row.get('compound', row.get('compound_id', ''))
                if compound in compound_list:
                    compounds_i.add(compound)
        
        for j, target_j in enumerate(target_list):
            if i != j:
                compounds_j = set()
                for _, row in validated_data.iterrows():
                    if row.get('target', row.get('target_id', '')) == target_j:
                        compound = row.get('compound', row.get('compound_id', ''))
                        if compound in compound_list:
                            compounds_j.add(compound)
                
                # Calculate Jaccard similarity
                intersection = compounds_i & compounds_j
                union = compounds_i | compounds_j
                
                if union:
                    protein_sim_matrix[i, j] = len(intersection) / len(union)
    
    # Calculate priorities for each compound
    compound_priorities = {}
    
    for i, compound_idx in enumerate(compound_indices):
        comp_id = reverse_node_map[compound_idx.item()]
        
        # Initialize target scores
        target_scores = {}
        
        for j, target_idx in enumerate(target_indices):
            target_id = reverse_node_map[target_idx.item()]
            
            # Calculate embedding similarity
            emb_sim = torch.cosine_similarity(
                embeddings[compound_idx].unsqueeze(0),
                embeddings[target_idx].unsqueeze(0)
            ).item()
            
            # Get target importance
            importance = target_importance.get(target_id, 0.5)
            adjusted_importance = np.power(importance, 1.5)  # Non-linear transformation
            
            # Calculate drug similarity component
            drug_sim_score = 0
            if weights['drug_sim'] > 0:
                for k, other_comp_id in enumerate(compound_list):
                    if other_comp_id != comp_id:
                        # Check if other compound interacts with this target
                        interacts = False
                        for _, row in validated_data.iterrows():
                            if (row.get('compound', row.get('compound_id', '')) == other_comp_id and 
                                row.get('target', row.get('target_id', '')) == target_id):
                                interacts = True
                                break
                        
                        if interacts:
                            drug_sim_score += drug_sim_matrix[i, k]
                
                # Normalize by number of compounds
                if len(compound_list) > 1:
                    drug_sim_score /= (len(compound_list) - 1)
            
            # Calculate protein similarity component
            protein_sim_score = 0
            if weights['protein_sim'] > 0:
                # Find targets that interact with this compound
                interacting_targets = []
                for _, row in validated_data.iterrows():
                    if row.get('compound', row.get('compound_id', '')) == comp_id:
                        target = row.get('target', row.get('target_id', ''))
                        if target in target_list and target != target_id:
                            interacting_targets.append(target_list.index(target))
                
                # Calculate similarity to interacting targets
                for k in interacting_targets:
                    protein_sim_score += protein_sim_matrix[j, k]
                
                # Normalize by number of interacting targets
                if interacting_targets:
                    protein_sim_score /= len(interacting_targets)
            
            # Calculate semantic similarity component
            semantic_sim_score = 0
            if weights['semantic'] > 0 and semantic_matrix is not None:
                semantic_sim_score = semantic_matrix[i, j]
            
            # Calculate weighted score
            weighted_score = (
                weights['embedding'] * emb_sim +
                weights['importance'] * adjusted_importance +
                weights['drug_sim'] * drug_sim_score +
                weights['protein_sim'] * protein_sim_score +
                weights['semantic'] * semantic_sim_score
            )
            
            # Store score
            target_scores[target_id] = weighted_score
        
        # Sort targets by score and keep top k
        sorted_targets = sorted(target_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        compound_priorities[comp_id] = dict(sorted_targets)
    
    return compound_priorities

def perform_multiple_runs(args, data, node_map, reverse_node_map, target_importance, validated_data, 
                         semantic_similarities=None, n_runs=5):
    """Perform multiple model runs for stability analysis"""
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create list to store predictions
    predictions_list = []
    
    # Get compound and target lists
    compound_list = [reverse_node_map[idx.item()] for idx in data.compound_indices]
    target_list = [reverse_node_map[idx.item()] for idx in data.target_indices]
    
    # Create weights dictionary
    weights = {
        'embedding': args.embedding_weight,
        'importance': args.importance_weight,
        'drug_sim': args.drug_sim_weight,
        'protein_sim': args.protein_sim_weight,
        'semantic': args.semantic_weight
    }
    
    # Perform multiple runs
    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}")
        
        # Create model
        model = RGCN(
            in_dim=data.x.shape[1],
            hidden_dim=256,
            out_dim=128,
            num_relations=data.num_relations,
            num_bases=4,
            dropout=0.3
        ).to(device)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            epochs=100,
            lr=0.0001,
            weight_decay=1e-5,
            margin=0.4,
            patience=20,
            device=device
        )
        
        # Train model
        data_device = data.to(device)
        embeddings = trainer.train(data_device)
        
        # Calculate priorities
        priorities = calculate_target_priorities(
            embeddings, node_map, reverse_node_map, data,
            target_importance, validated_data, semantic_similarities,
            weights=weights
        )
        
        # Store predictions
        predictions_list.append(priorities)
    
    return predictions_list, compound_list, target_list

def main():
    """Main function"""
    # Parse command line arguments
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    kg_data, validated_data, target_importance = load_data(args)
    
    # Build semantic graph
    data, node_map, reverse_node_map, relation_types = build_semantic_graph(kg_data)
    
    # Load semantic similarities if available
    semantic_similarities_path = 'results/semantic_analysis/semantic_similarities.csv'
    semantic_similarities = None
    if os.path.exists(semantic_similarities_path) and args.semantic_weight > 0:
        print(f"Loading semantic similarities from {semantic_similarities_path}")
        semantic_similarities = pd.read_csv(semantic_similarities_path)
    
    # Load or train model
    model, embeddings = load_or_train_model(args, data)
    
    # Get compound and target lists
    compound_list = [reverse_node_map[idx.item()] for idx in data.compound_indices]
    target_list = [reverse_node_map[idx.item()] for idx in data.target_indices]
    
    # Create weights dictionary
    weights = {
        'embedding': args.embedding_weight,
        'importance': args.importance_weight,
        'drug_sim': args.drug_sim_weight,
        'protein_sim': args.protein_sim_weight,
        'semantic': args.semantic_weight
    }
    
    # Calculate priorities
    priorities = calculate_target_priorities(
        embeddings, node_map, reverse_node_map, data,
        target_importance, validated_data, semantic_similarities,
        weights=weights
    )
    
    # Validate model
    validation_output_path = os.path.join(args.output_dir, 'validation_results.png')
    metrics = validate_model(
        priorities, validated_data, compound_list, target_list, validation_output_path
    )
    
    # Print metrics
    print("\nModel Validation Results:")
    print(f"  Average Precision: {metrics.get('average_precision', 0):.4f}")
    print(f"  ROC AUC: {metrics.get('roc_auc', 0):.4f}")
    print(f"  MRR: {metrics.get('mrr', 0):.4f}")
    print(f"  Hit@5: {metrics['hit_rates'].get('hit@5', 0):.4f}")
    print(f"  Hit@10: {metrics['hit_rates'].get('hit@10', 0):.4f}")
    print(f"  Hit@20: {metrics['hit_rates'].get('hit@20', 0):.4f}")
    
    # Perform leave-one-out cross-validation if requested
    if args.loocv:
        loocv_results = leave_one_out_validation(
            model, data, 
            lambda emb, nm, rnm, vd: calculate_target_priorities(
                emb, node_map, reverse_node_map, data,
                target_importance, vd, semantic_similarities,
                weights=weights
            ), 
            validated_data, compound_list, target_list
        )
    
    # Perform k-fold cross-validation if requested
    if args.kfold > 1:
        kfold_results, avg_metrics, std_metrics = k_fold_cross_validation(
            model, data, 
            lambda emb, nm, rnm, vd: calculate_target_priorities(
                emb, node_map, reverse_node_map, data,
                target_importance, vd, semantic_similarities,
                weights=weights
            ), 
            validated_data, compound_list, target_list,
            k=args.kfold
        )
    
    # Perform clustering analysis if requested
    if args.clustering:
        clustering_results = analyze_embedding_space(
            embeddings, node_map, reverse_node_map, compound_list, target_list
        )
    
    # Perform stability analysis if requested
    if args.stability > 1:
        predictions_list, compound_list, target_list = perform_multiple_runs(
            args, data, node_map, reverse_node_map, 
            target_importance, validated_data, semantic_similarities,
            n_runs=args.stability
        )
        
        stability_results = analyze_stability(
            predictions_list, compound_list, target_list
        )
    
    # Perform target overlap analysis if requested
    if args.overlap > 1:
        predictions_list, compound_list, target_list = perform_multiple_runs(
            args, data, node_map, reverse_node_map, 
            target_importance, validated_data, semantic_similarities,
            n_runs=args.overlap
        )
        
        overlap_results, summary_df = calculate_top_n_overlap(
            predictions_list, compound_list
        )
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Enhanced TCM Target Prioritization Evaluation')
    
    # Input data options
    parser.add_argument('--kg_data', type=str, default='data/raw/kg_data_extended.csv',
                      help='Knowledge graph data file')
    parser.add_argument('--validated_data', type=str, default='data/raw/validated_interactions.csv',
                      help='Validated interactions file')
    parser.add_argument('--disease_importance', type=str, default='data/raw/disease_importance_extended.csv',
                      help='Disease importance data file')
    
    # Model options
    parser.add_argument('--model_path', type=str, default='results/models/rgcn_model.pt',
                      help='Trained model file')
    parser.add_argument('--embeddings_path', type=str, default='results/embeddings/node_embeddings.pt',
                      help='Pre-computed embeddings file')
    
    # Evaluation options
    parser.add_argument('--loocv', action='store_true',
                      help='Perform leave-one-out cross-validation')
    parser.add_argument('--kfold', type=int, default=0,
                      help='Perform k-fold cross-validation with specified k')
    parser.add_argument('--clustering', action='store_true',
                      help='Perform embedding space clustering analysis')
    parser.add_argument('--stability', type=int, default=0,
                      help='Perform stability analysis with specified number of runs')
    parser.add_argument('--overlap', type=int, default=0,
                      help='Perform target overlap analysis with specified number of runs')
    
    # Weight parameters for prioritization
    parser.add_argument('--embedding_weight', type=float, default=0.05,
                      help='Weight for embedding similarity (default: 0.05)')
    parser.add_argument('--importance_weight', type=float, default=0.8,
                      help='Weight for target importance (default: 0.8)')
    parser.add_argument('--drug_sim_weight', type=float, default=0.075,
                      help='Weight for drug similarity (default: 0.075)')
    parser.add_argument('--protein_sim_weight', type=float, default=0.075,
                      help='Weight for protein similarity (default: 0.075)')
    parser.add_argument('--semantic_weight', type=float, default=0.0,
                      help='Weight for semantic similarity (default: 0.0)')
    
    # Other options
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                      help='Output directory')
    
    # Run main function
    main()
