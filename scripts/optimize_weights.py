#!/usr/bin/env python
"""
Weight optimization script for enhanced TCM target prioritization

This script evaluates different combinations of weights for the four components:
1. Embedding similarity (from graph neural network)
2. Target importance (from disease relevance)
3. Drug structural similarity 
4. Protein sequence similarity

It finds the optimal weights based on multiple evaluation metrics.
"""

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
    compound_indices = [idx for id_name, idx in node_map.items() 
                       if isinstance(id_name, str) and id_name in validated_data['compound'].unique()]
    target_indices = [idx for id_name, idx in node_map.items() 
                     if isinstance(id_name, str) and id_name in validated_data['target'].unique()]
    
    all_target_indices = [idx for id_name, idx in node_map.items() 
                         if isinstance(id_name, str) and id_name in target_importance]
    
    # Calculate priorities with the given weights
    priorities = calculate_priorities_with_similarity(
        embeddings, node_map, reverse_node_map,
        torch.tensor(compound_indices), 
        torch.tensor(all_target_indices),
        target_importance, drug_similarity, protein_similarity,
        embedding_weight, importance_weight, drug_sim_weight, protein_sim_weight
    )
    
    # Create validation matrix
    compound_list = [reverse_node_map[idx] for idx in compound_indices]
    target_list = [reverse_node_map[idx] for idx in all_target_indices]
    
    validation_matrix = np.zeros((len(compound_list), len(target_list)))
    
    # Mapping from IDs to indices
    compound_to_idx = {comp_id: i for i, comp_id in enumerate(compound_list)}
    target_to_idx = {target_id: i for i, target_id in enumerate(target_list)}
    
    # Fill matrix with validated interactions
    for _, row in validated_data.iterrows():
        comp_id = row['compound']
        target_id = row['target']
        
        if comp_id in compound_to_idx and target_id in target_to_idx:
            comp_idx = compound_to_idx[comp_id]
            target_idx = target_to_idx[target_id]
            validation_matrix[comp_idx, target_idx] = 1
    
    # Create prediction score matrix
    prediction_matrix = np.zeros((len(compound_list), len(target_list)))
    
    for comp_id, target_scores in priorities.items():
        if comp_id in compound_to_idx:
            comp_idx = compound_to_idx[comp_id]
            for target_id, score in target_scores.items():
                if target_id in target_to_idx:
                    target_idx = target_to_idx[target_id]
                    prediction_matrix[comp_idx, target_idx] = score
    
    # Calculate metrics
    y_true = validation_matrix.flatten()
    y_pred = prediction_matrix.flatten()
    
    # Average precision score
    ap_score = average_precision_score(y_true, y_pred)
    
    # ROC AUC
    auroc = roc_auc_score(y_true, y_pred)
    
    # Calculate Hit@k and MRR
    hit_at_5 = 0
    hit_at_10 = 0
    hit_at_20 = 0
    reciprocal_ranks = []
    
    for comp_idx, comp_id in enumerate(compound_list):
        if comp_id in priorities:
            # Get validated targets for this compound
            valid_targets = [target_list[i] for i in np.where(validation_matrix[comp_idx] == 1)[0]]
            
            if valid_targets:
                # Get predicted rankings
                pred_targets = list(priorities[comp_id].keys())
                
                # Calculate metrics
                for valid_target in valid_targets:
                    if valid_target in pred_targets:
                        rank = pred_targets.index(valid_target) + 1
                        reciprocal_ranks.append(1.0/rank)
                        
                        # Hit@k
                        if rank <= 5:
                            hit_at_5 += 1
                        if rank <= 10:
                            hit_at_10 += 1
                        if rank <= 20:
                            hit_at_20 += 1
    
    # Normalize hit rates
    total_validated = np.sum(validation_matrix)
    hit_at_5 = hit_at_5 / total_validated if total_validated > 0 else 0
    hit_at_10 = hit_at_10 / total_validated if total_validated > 0 else 0
    hit_at_20 = hit_at_20 / total_validated if total_validated > 0 else 0
    
    # Mean Reciprocal Rank
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    
    return {
        'embedding_weight': embedding_weight,
        'importance_weight': importance_weight,
        'drug_sim_weight': drug_sim_weight,
        'protein_sim_weight': protein_sim_weight,
        'ap_score': ap_score,
        'auroc': auroc,
        'hit_at_5': hit_at_5,
        'hit_at_10': hit_at_10,
        'hit_at_20': hit_at_20,
        'mrr': mrr
    }

def optimize_weights():
    """Run weight optimization with similarity components"""
    print("Starting weight optimization for enhanced prioritization...")
    
    # Create results directory
    os.makedirs('results/weight_optimization', exist_ok=True)
    
    # Train model to get embeddings or load if already exist
    if os.path.exists('results/embeddings/node_embeddings.pt') and \
       os.path.exists('data/processed/node_map.pt') and \
       os.path.exists('data/processed/reverse_node_map.pt'):
        print("Loading existing embeddings...")
        embeddings = torch.load('results/embeddings/node_embeddings.pt')
        node_map = torch.load('data/processed/node_map.pt')
        reverse_node_map = torch.load('data/processed/reverse_node_map.pt')
        _, _, target_importance, validated_data, drug_similarity, protein_similarity = load_data()
    else:
        print("Training model to generate embeddings...")
        embeddings, node_map, reverse_node_map, target_importance, drug_similarity, protein_similarity = train_model_with_embeddings()
        _, _, _, validated_data, _, _ = load_data()
    
    # Define weight ranges to search
    # We now have 4 weights that need to sum to 1.0
    embedding_weights = [0.05, 0.1, 0.2, 0.3]
    importance_weights = [0.4, 0.6, 0.7, 0.8]
    drug_sim_weights = [0.0, 0.05, 0.1, 0.15]
    protein_sim_weights = [0.0, 0.05, 0.1, 0.15]
    
    # Store results
    results = []
    
    # Grid search all combinations that approximately sum to 1.0
    print("Evaluating weight combinations...")
    for emb_w, imp_w, drug_w, prot_w in tqdm(list(product(
        embedding_weights, importance_weights, drug_sim_weights, protein_sim_weights))):
        
        # Skip if weights don't approximately sum to 1.0
        # Allow small rounding errors
        total = emb_w + imp_w + drug_w + prot_w
        if abs(total - 1.0) > 0.05:
            continue
        
        # Evaluate this weight combination
        metrics = evaluate_weights(
            embeddings, node_map, reverse_node_map,
            target_importance, validated_data, 
            drug_similarity, protein_similarity,
            emb_w, imp_w, drug_w, prot_w
        )
        
        results.append(metrics)
        print(f"Weights: emb={emb_w:.2f}, imp={imp_w:.2f}, drug={drug_w:.2f}, prot={prot_w:.2f}, "
              f"MRR: {metrics['mrr']:.4f}, AP: {metrics['ap_score']:.4f}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv('results/weight_optimization/enhanced_weight_results.csv', index=False)
    
    # Plot results
    plot_enhanced_weight_optimization(results_df)
    
    # Find optimal weights
    best_ap_idx = results_df['ap_score'].idxmax()
    best_mrr_idx = results_df['mrr'].idxmax()
    best_hit10_idx = results_df['hit_at_10'].idxmax()
    
    best_ap_weights = {
        'embedding_weight': results_df.iloc[best_ap_idx]['embedding_weight'],
        'importance_weight': results_df.iloc[best_ap_idx]['importance_weight'],
        'drug_sim_weight': results_df.iloc[best_ap_idx]['drug_sim_weight'],
        'protein_sim_weight': results_df.iloc[best_ap_idx]['protein_sim_weight']
    }
    
    best_mrr_weights = {
        'embedding_weight': results_df.iloc[best_mrr_idx]['embedding_weight'],
        'importance_weight': results_df.iloc[best_mrr_idx]['importance_weight'],
        'drug_sim_weight': results_df.iloc[best_mrr_idx]['drug_sim_weight'],
        'protein_sim_weight': results_df.iloc[best_mrr_idx]['protein_sim_weight']
    }
    
    best_hit10_weights = {
        'embedding_weight': results_df.iloc[best_hit10_idx]['embedding_weight'],
        'importance_weight': results_df.iloc[best_hit10_idx]['importance_weight'],
        'drug_sim_weight': results_df.iloc[best_hit10_idx]['drug_sim_weight'],
        'protein_sim_weight': results_df.iloc[best_hit10_idx]['protein_sim_weight']
    }
    
    print("\n=== Optimal Weight Configurations ===")
    print(f"Best weights for Average Precision: {best_ap_weights}")
    print(f"Best weights for MRR: {best_mrr_weights}")
    print(f"Best weights for Hit@10: {best_hit10_weights}")
    
    # Save optimal weights
    with open('results/weight_optimization/enhanced_optimal_weights.txt', 'w') as f:
        f.write("=== Optimal Weight Configurations ===\n\n")
        f.write(f"Best weights for Average Precision:\n")
        f.write(f"  Embedding Weight: {best_ap_weights['embedding_weight']:.2f}\n")
        f.write(f"  Importance Weight: {best_ap_weights['importance_weight']:.2f}\n")
        f.write(f"  Drug Similarity Weight: {best_ap_weights['drug_sim_weight']:.2f}\n")
        f.write(f"  Protein Similarity Weight: {best_ap_weights['protein_sim_weight']:.2f}\n\n")
        
        f.write(f"Best weights for Mean Reciprocal Rank:\n")
        f.write(f"  Embedding Weight: {best_mrr_weights['embedding_weight']:.2f}\n")
        f.write(f"  Importance Weight: {best_mrr_weights['importance_weight']:.2f}\n")
        f.write(f"  Drug Similarity Weight: {best_mrr_weights['drug_sim_weight']:.2f}\n")
        f.write(f"  Protein Similarity Weight: {best_mrr_weights['protein_sim_weight']:.2f}\n\n")
        
        f.write(f"Best weights for Hit@10:\n")
        f.write(f"  Embedding Weight: {best_hit10_weights['embedding_weight']:.2f}\n")
        f.write(f"  Importance Weight: {best_hit10_weights['importance_weight']:.2f}\n")
        f.write(f"  Drug Similarity Weight: {best_hit10_weights['drug_sim_weight']:.2f}\n")
        f.write(f"  Protein Similarity Weight: {best_hit10_weights['protein_sim_weight']:.2f}\n\n")
        
        f.write(f"For most use cases, the MRR-optimized weights are recommended:\n")
        f.write(f"  --embedding_weight {best_mrr_weights['embedding_weight']:.2f} \\\n")
        f.write(f"  --importance_weight {best_mrr_weights['importance_weight']:.2f} \\\n")
        f.write(f"  --drug_sim_weight {best_mrr_weights['drug_sim_weight']:.2f} \\\n")
        f.write(f"  --protein_sim_weight {best_mrr_weights['protein_sim_weight']:.2f}\n")
    
    return best_mrr_weights

def plot_enhanced_weight_optimization(results_df):
    """Create visualization of enhanced weight optimization results"""
    plt.figure(figsize=(18, 15))
    
    # Plot MRR heatmap
    plt.subplot(2, 2, 1)
    plot_weight_heatmap(results_df, 'mrr', 'Mean Reciprocal Rank (MRR)')
    
    # Plot AP heatmap
    plt.subplot(2, 2, 2)
    plot_weight_heatmap(results_df, 'ap_score', 'Average Precision')
    
    # Plot Hit@10 heatmap
    plt.subplot(2, 2, 3)
    plot_weight_heatmap(results_df, 'hit_at_10', 'Hit@10 Rate')
    
    # Plot weights importance
    plt.subplot(2, 2, 4)
    plot_weight_importance(results_df)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('results/weight_optimization/enhanced_weight_optimization.png', dpi=300)
    plt.close()
    print("Enhanced weight optimization plot saved to results/weight_optimization/enhanced_weight_optimization.png")

def plot_weight_heatmap(results_df, metric_col, title):
    """
    Create heatmap to visualize how weight combinations affect a metric
    
    Args:
        results_df: DataFrame with weight optimization results
        metric_col: Column name for the metric to visualize
        title: Plot title
    """
    # Group by embedding and importance weights
    grouped = results_df.groupby(['embedding_weight', 'importance_weight'])[metric_col].mean().reset_index()
    
    # Create a pivot table for heatmap
    pivot_table = grouped.pivot(index='importance_weight', columns='embedding_weight', values=metric_col)
    
    # Draw heatmap
    import seaborn as sns
    ax = sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.3f')
    plt.title(f'{title} by Weight Combination')
    plt.xlabel('Embedding Weight')
    plt.ylabel('Importance Weight')

def plot_weight_importance(results_df):
    """
    Plot the relative importance of each weight component
    
    Args:
        results_df: DataFrame with weight optimization results
    """
    # Find top performing configurations
    top_n = 10
    top_rows = results_df.nlargest(top_n, 'mrr')
    
    # Average weights from top configurations
    avg_weights = {
        'Embedding': top_rows['embedding_weight'].mean(),
        'Importance': top_rows['importance_weight'].mean(),
        'Drug Similarity': top_rows['drug_sim_weight'].mean(),
        'Protein Similarity': top_rows['protein_sim_weight'].mean()
    }
    
    # Draw bar chart
    plt.bar(avg_weights.keys(), avg_weights.values())
    plt.title(f'Average Weight Values in Top {top_n} Configurations')
    plt.ylabel('Average Weight Value')
    plt.ylim(0, 1.0)
    
    # Add value labels on bars
    for i, (k, v) in enumerate(avg_weights.items()):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center')

def plot_weight_scatter(results_df):
    """
    Create scatter plots to show relationships between weights and metrics
    
    Args:
        results_df: DataFrame with weight optimization results
    """
    # Create a figure with 3 subplots
    plt.figure(figsize=(18, 6))
    
    # Metrics to visualize
    metrics = {
        'mrr': 'Mean Reciprocal Rank',
        'ap_score': 'Average Precision',
        'hit_at_10': 'Hit@10 Rate'
    }
    
    # Create a scatter plot for each metric
    for i, (metric_col, metric_name) in enumerate(metrics.items(), 1):
        plt.subplot(1, 3, i)
        
        # Scatter plot with two color dimensions
        sc = plt.scatter(
            results_df['embedding_weight'], 
            results_df['importance_weight'],
            c=results_df[metric_col], 
            s=100 * (results_df['drug_sim_weight'] + results_df['protein_sim_weight'] + 0.1),
            alpha=0.7,
            cmap='viridis'
        )
        
        # Add colorbar
        cbar = plt.colorbar(sc)
        cbar.set_label(metric_name)
        
        # Plot settings
        plt.title(f'{metric_name} by Weight Combination')
        plt.xlabel('Embedding Weight')
        plt.ylabel('Importance Weight')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Annotate top points
        top_idx = results_df[metric_col].idxmax()
        top_point = results_df.iloc[top_idx]
        plt.annotate(
            f"Best: {top_point[metric_col]:.4f}",
            (top_point['embedding_weight'], top_point['importance_weight']),
            xytext=(10, 10),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='red')
        )
    
    plt.tight_layout()
    plt.savefig('results/weight_optimization/weight_scatter_plots.png', dpi=300)
    plt.close()

def perform_sensitivity_analysis(
    embeddings, node_map, reverse_node_map,
    target_importance, validated_data, drug_similarity, protein_similarity,
    base_weights):
    """
    Perform sensitivity analysis by varying each weight individually
    
    Args:
        embeddings: Node embeddings
        node_map: Mapping from node IDs to indices
        reverse_node_map: Mapping from indices to node IDs
        target_importance: Dictionary of target importance scores
        validated_data: DataFrame with validated interactions
        drug_similarity: Drug similarity matrix
        protein_similarity: Protein similarity matrix
        base_weights: Dictionary of base weight values
        
    Returns:
        DataFrame with sensitivity analysis results
    """
    print("Performing sensitivity analysis...")
    
    # Extract base weights
    base_emb_w = base_weights['embedding_weight']
    base_imp_w = base_weights['importance_weight']
    base_drug_w = base_weights['drug_sim_weight']
    base_prot_w = base_weights['protein_sim_weight']
    
    # Perturbation range
    perturbations = np.linspace(-0.2, 0.2, 9)  # -0.2, -0.15, -0.1, ..., 0.2
    
    # Store results
    sensitivity_results = []
    
    # Base performance (no perturbation)
    base_metrics = evaluate_weights(
        embeddings, node_map, reverse_node_map,
        target_importance, validated_data, 
        drug_similarity, protein_similarity,
        base_emb_w, base_imp_w, base_drug_w, base_prot_w
    )
    
    print(f"Base performance - MRR: {base_metrics['mrr']:.4f}, AP: {base_metrics['ap_score']:.4f}")
    
    # Vary embedding weight
    for pert in perturbations:
        # Skip if perturbation would make weight negative
        if base_emb_w + pert < 0:
            continue
            
        # Adjust weights to maintain sum of 1.0
        # Distribute the perturbation proportionally to other weights
        total_other = base_imp_w + base_drug_w + base_prot_w
        
        if total_other > 0:
            factor = (total_other - pert) / total_other
            
            # New weights
            new_emb_w = base_emb_w + pert
            new_imp_w = base_imp_w * factor
            new_drug_w = base_drug_w * factor
            new_prot_w = base_prot_w * factor
            
            # Evaluate
            metrics = evaluate_weights(
                embeddings, node_map, reverse_node_map,
                target_importance, validated_data, 
                drug_similarity, protein_similarity,
                new_emb_w, new_imp_w, new_drug_w, new_prot_w
            )
            
            # Add to results
            metrics['perturbed_weight'] = 'embedding'
            metrics['perturbation'] = pert
            sensitivity_results.append(metrics)
    
    # Vary importance weight
    for pert in perturbations:
        if base_imp_w + pert < 0:
            continue
            
        total_other = base_emb_w + base_drug_w + base_prot_w
        
        if total_other > 0:
            factor = (total_other - pert) / total_other
            
            new_imp_w = base_imp_w + pert
            new_emb_w = base_emb_w * factor
            new_drug_w = base_drug_w * factor
            new_prot_w = base_prot_w * factor
            
            metrics = evaluate_weights(
                embeddings, node_map, reverse_node_map,
                target_importance, validated_data, 
                drug_similarity, protein_similarity,
                new_emb_w, new_imp_w, new_drug_w, new_prot_w
            )
            
            metrics['perturbed_weight'] = 'importance'
            metrics['perturbation'] = pert
            sensitivity_results.append(metrics)
    
    # Vary drug similarity weight
    for pert in perturbations:
        if base_drug_w + pert < 0:
            continue
            
        total_other = base_emb_w + base_imp_w + base_prot_w
        
        if total_other > 0:
            factor = (total_other - pert) / total_other
            
            new_drug_w = base_drug_w + pert
            new_emb_w = base_emb_w * factor
            new_imp_w = base_imp_w * factor
            new_prot_w = base_prot_w * factor
            
            metrics = evaluate_weights(
                embeddings, node_map, reverse_node_map,
                target_importance, validated_data, 
                drug_similarity, protein_similarity,
                new_emb_w, new_imp_w, new_drug_w, new_prot_w
            )
            
            metrics['perturbed_weight'] = 'drug_similarity'
            metrics['perturbation'] = pert
            sensitivity_results.append(metrics)
    
    # Vary protein similarity weight
    for pert in perturbations:
        if base_prot_w + pert < 0:
            continue
            
        total_other = base_emb_w + base_imp_w + base_drug_w
        
        if total_other > 0:
            factor = (total_other - pert) / total_other
            
            new_prot_w = base_prot_w + pert
            new_emb_w = base_emb_w * factor
            new_imp_w = base_imp_w * factor
            new_drug_w = base_drug_w * factor
            
            metrics = evaluate_weights(
                embeddings, node_map, reverse_node_map,
                target_importance, validated_data, 
                drug_similarity, protein_similarity,
                new_emb_w, new_imp_w, new_drug_w, new_prot_w
            )
            
            metrics['perturbed_weight'] = 'protein_similarity'
            metrics['perturbation'] = pert
            sensitivity_results.append(metrics)
    
    # Convert to DataFrame
    sensitivity_df = pd.DataFrame(sensitivity_results)
    
    # Save results
    sensitivity_df.to_csv('results/weight_optimization/sensitivity_analysis.csv', index=False)
    
    # Plot sensitivity
    plot_sensitivity_analysis(sensitivity_df, base_metrics)
    
    return sensitivity_df

def plot_sensitivity_analysis(sensitivity_df, base_metrics):
    """
    Plot sensitivity analysis results
    
    Args:
        sensitivity_df: DataFrame with sensitivity analysis results
        base_metrics: Dictionary with base performance metrics
    """
    plt.figure(figsize=(15, 10))
    
    # Metrics to plot
    metrics = {
        'mrr': 'Mean Reciprocal Rank',
        'ap_score': 'Average Precision',
        'hit_at_10': 'Hit@10 Rate'
    }
    
    # Plot each metric
    for i, (metric_col, metric_name) in enumerate(metrics.items(), 1):
        plt.subplot(2, 2, i)
        
        # Get base performance for this metric
        base_value = base_metrics[metric_col]
        
        # Plot line for each weight
        for weight in ['embedding', 'importance', 'drug_similarity', 'protein_similarity']:
            # Filter data for this weight
            weight_data = sensitivity_df[sensitivity_df['perturbed_weight'] == weight]
            
            if len(weight_data) > 0:
                # Sort by perturbation for proper line
                weight_data = weight_data.sort_values('perturbation')
                
                # Calculate percent change from base
                pct_change = (weight_data[metric_col] / base_value - 1) * 100
                
                # Plot line
                plt.plot(
                    weight_data['perturbation'],
                    pct_change,
                    'o-',
                    label=f"{weight.replace('_', ' ').title()}"
                )
        
        # Plot settings
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        plt.title(f'Sensitivity Analysis: {metric_name}')
        plt.xlabel('Weight Perturbation')
        plt.ylabel('Percent Change (%)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
    
    # Plot weight contribution
    plt.subplot(2, 2, 4)
    
    # Calculate slope of each weight's effect
    slopes = {}
    for weight in ['embedding', 'importance', 'drug_similarity', 'protein_similarity']:
        # Filter data for this weight
        weight_data = sensitivity_df[sensitivity_df['perturbed_weight'] == weight]
        
        if len(weight_data) > 1:
            # Filter to small perturbations for linearity
            small_pert = weight_data[abs(weight_data['perturbation']) <= 0.1]
            
            if len(small_pert) > 1:
                # Calculate slope for MRR
                x = small_pert['perturbation'].values
                y = small_pert['mrr'].values
                
                # Simple linear regression slope
                slope = np.polyfit(x, y, 1)[0]
                slopes[weight] = abs(slope)
    
    # Plot barplot of absolute slopes
    if slopes:
        # Normalize to sum to 1
        total = sum(slopes.values())
        if total > 0:
            norm_slopes = {k: v/total for k, v in slopes.items()}
            
            # Plot
            plt.bar(norm_slopes.keys(), norm_slopes.values())
            plt.title('Relative Importance of Weights')
            plt.ylabel('Normalized Impact')
            plt.ylim(0, 1.0)
            
            # Add value labels
            for i, (k, v) in enumerate(norm_slopes.items()):
                plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('results/weight_optimization/sensitivity_analysis.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    # Run weight optimization
    best_weights = optimize_weights()
    
    # Create embeddings path for sensitivity analysis
    if os.path.exists('results/embeddings/node_embeddings.pt') and \
       os.path.exists('data/processed/node_map.pt') and \
       os.path.exists('data/processed/reverse_node_map.pt'):
        print("\nRunning sensitivity analysis on optimal weights...")
        
        # Load embeddings
        embeddings = torch.load('results/embeddings/node_embeddings.pt')
        node_map = torch.load('data/processed/node_map.pt')
        reverse_node_map = torch.load('data/processed/reverse_node_map.pt')
        _, _, target_importance, validated_data, drug_similarity, protein_similarity = load_data()
        
        # Run sensitivity analysis
        perform_sensitivity_analysis(
            embeddings, node_map, reverse_node_map,
            target_importance, validated_data,
            drug_similarity, protein_similarity,
            best_weights
        )
    
    # Create additional visualizations
    if os.path.exists('results/weight_optimization/enhanced_weight_results.csv'):
        results_df = pd.read_csv('results/weight_optimization/enhanced_weight_results.csv')
        plot_weight_scatter(results_df)
    
    print("\nWeight optimization completed!")
