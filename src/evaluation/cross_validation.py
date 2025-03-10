"""
Cross-validation module for TCM target prioritization

This module provides leave-one-out and k-fold cross-validation methods
for evaluating model performance.
"""
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold

from src.evaluation.metrics import calculate_average_precision, calculate_roc_auc, calculate_hit_rates, calculate_mrr

def leave_one_out_validation(model, data, priorities_func, validated_data, compound_list, target_list):
    """
    Perform leave-one-out cross-validation
    
    Args:
        model: Trained model
        data: Graph data object
        priorities_func: Function for calculating target priorities
        validated_data: DataFrame with validated interactions
        compound_list: List of compound IDs
        target_list: List of target IDs
        
    Returns:
        DataFrame with cross-validation results
    """
    print("Performing leave-one-out cross-validation...")
    
    # Extract validated interactions
    validation_pairs = []
    for _, row in validated_data.iterrows():
        compound_id = row.get('compound', row.get('compound_id', None))
        target_id = row.get('target', row.get('target_id', None))
        if compound_id in compound_list and target_id in target_list:
            validation_pairs.append((compound_id, target_id))
    
    # Create maps from IDs to indices
    compound_to_idx = {comp_id: idx for idx, comp_id in enumerate(compound_list)}
    target_to_idx = {target_id: idx for idx, target_id in enumerate(target_list)}
    
    # Initialize results
    results = []
    
    # Perform leave-one-out validation
    for comp_id, target_id in tqdm(validation_pairs, desc="LOOCV"):
        # Create validation matrix with this pair left out
        loocv_data = validated_data.copy()
        mask = ~((loocv_data['compound'] == comp_id) & (loocv_data['target'] == target_id))
        train_data = loocv_data[mask]
        
        # Get embeddings from model
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index, data.edge_type, 
                               edge_weight=data.edge_weight if hasattr(data, 'edge_weight') else None)
        
        # Calculate priorities without this pair
        priorities = priorities_func(embeddings, None, None, train_data)
        
        # Check rank of left-out pair
        if comp_id in priorities:
            target_scores = priorities[comp_id]
            if target_id in target_scores:
                rank = list(target_scores.keys()).index(target_id) + 1
                reciprocal_rank = 1.0 / rank
                
                # Add to results
                results.append({
                    'compound_id': comp_id,
                    'target_id': target_id,
                    'rank': rank,
                    'reciprocal_rank': reciprocal_rank,
                    'score': target_scores[target_id],
                    'in_top_5': rank <= 5,
                    'in_top_10': rank <= 10,
                    'in_top_20': rank <= 20
                })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    mrr = results_df['reciprocal_rank'].mean()
    hit_5 = results_df['in_top_5'].mean()
    hit_10 = results_df['in_top_10'].mean()
    hit_20 = results_df['in_top_20'].mean()
    
    print(f"LOOCV Results - MRR: {mrr:.4f}, Hit@5: {hit_5:.4f}, Hit@10: {hit_10:.4f}, Hit@20: {hit_20:.4f}")
    
    # Save results
    os.makedirs('results/evaluation', exist_ok=True)
    results_df.to_csv('results/evaluation/loocv_results.csv', index=False)
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    plt.hist(results_df['rank'], bins=20, alpha=0.7)
    plt.axvline(x=5, color='r', linestyle='--', label='Top 5')
    plt.axvline(x=10, color='g', linestyle='--', label='Top 10')
    plt.axvline(x=20, color='b', linestyle='--', label='Top 20')
    plt.xlabel('Rank')
    plt.ylabel('Count')
    plt.title('Leave-One-Out Cross-Validation Rank Distribution')
    plt.legend()
    plt.savefig('results/evaluation/loocv_ranks.png', dpi=300)
    plt.close()
    
    return results_df

def k_fold_cross_validation(model, data, priorities_func, validated_data, compound_list, target_list, k=5):
    """
    Perform k-fold cross-validation
    
    Args:
        model: Trained model
        data: Graph data object
        priorities_func: Function for calculating target priorities
        validated_data: DataFrame with validated interactions
        compound_list: List of compound IDs
        target_list: List of target IDs
        k: Number of folds
        
    Returns:
        DataFrame with cross-validation results
    """
    print(f"Performing {k}-fold cross-validation...")
    
    # Create validation matrix
    validation_matrix = np.zeros((len(compound_list), len(target_list)))
    
    # Create maps from IDs to indices
    compound_to_idx = {comp_id: idx for idx, comp_id in enumerate(compound_list)}
    target_to_idx = {target_id: idx for idx, target_id in enumerate(target_list)}
    
    # Fill validation matrix
    for _, row in validated_data.iterrows():
        compound_id = row.get('compound', row.get('compound_id', None))
        target_id = row.get('target', row.get('target_id', None))
        
        if compound_id in compound_to_idx and target_id in target_to_idx:
            comp_idx = compound_to_idx[compound_id]
            target_idx = target_to_idx[target_id]
            validation_matrix[comp_idx, target_idx] = 1
    
    # Prepare folds
    validated_pairs = []
    for i, comp_id in enumerate(compound_list):
        for j, target_id in enumerate(target_list):
            if validation_matrix[i, j] == 1:
                validated_pairs.append((i, j, comp_id, target_id))
    
    # Initialize K-fold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Initialize results
    fold_results = []
    metrics = {
        'ap': [],
        'auc': [],
        'mrr': [],
        'hit@5': [],
        'hit@10': [],
        'hit@20': []
    }
    
    # Perform k-fold cross-validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(validated_pairs)):
        print(f"Fold {fold+1}/{k}")
        
        # Create train and test data
        train_pairs = [validated_pairs[i] for i in train_idx]
        test_pairs = [validated_pairs[i] for i in test_idx]
        
        # Create training validation matrix
        train_validation = np.zeros((len(compound_list), len(target_list)))
        for i, j, _, _ in train_pairs:
            train_validation[i, j] = 1
        
        # Create test validation matrix
        test_validation = np.zeros((len(compound_list), len(target_list)))
        for i, j, _, _ in test_pairs:
            test_validation[i, j] = 1
        
        # Get embeddings from model
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index, data.edge_type, 
                               edge_weight=data.edge_weight if hasattr(data, 'edge_weight') else None)
        
        # Create train data for priorities function
        train_validated_data = pd.DataFrame([
            {'compound': comp_id, 'target': target_id}
            for _, _, comp_id, target_id in train_pairs
        ])
        
        # Calculate priorities using training data
        priorities = priorities_func(embeddings, None, None, train_validated_data)
        
        # Create prediction matrix
        prediction_matrix = np.zeros((len(compound_list), len(target_list)))
        for i, comp_id in enumerate(compound_list):
            if comp_id in priorities:
                for target_id, score in priorities[comp_id].items():
                    if target_id in target_to_idx:
                        j = target_to_idx[target_id]
                        prediction_matrix[i, j] = score
        
        # Calculate metrics for this fold
        ap = calculate_average_precision(test_validation.flatten(), prediction_matrix.flatten())
        auc = calculate_roc_auc(test_validation.flatten(), prediction_matrix.flatten())
        hit_rates = calculate_hit_rates(test_validation, prediction_matrix)
        mrr_value = calculate_mrr(test_validation, prediction_matrix)
        
        # Store metrics
        metrics['ap'].append(ap)
        metrics['auc'].append(auc)
        metrics['mrr'].append(mrr_value)
        metrics['hit@5'].append(hit_rates['hit@5'])
        metrics['hit@10'].append(hit_rates['hit@10'])
        metrics['hit@20'].append(hit_rates['hit@20'])
        
        # Store fold results
        fold_results.append({
            'fold': fold + 1,
            'ap': ap,
            'auc': auc,
            'mrr': mrr_value,
            'hit@5': hit_rates['hit@5'],
            'hit@10': hit_rates['hit@10'],
            'hit@20': hit_rates['hit@20']
        })
    
    # Calculate average metrics
    avg_metrics = {metric: np.mean(values) for metric, values in metrics.items()}
    std_metrics = {metric: np.std(values) for metric, values in metrics.items()}
    
    # Print results
    print(f"\n{k}-Fold Cross-Validation Results:")
    for metric, value in avg_metrics.items():
        print(f"  {metric}: {value:.4f} ± {std_metrics[metric]:.4f}")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(fold_results)
    summary_df.to_csv('results/evaluation/kfold_results.csv', index=False)
    
    # Plot metrics
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(['ap', 'auc', 'mrr', 'hit@10']):
        plt.subplot(2, 2, i+1)
        plt.bar(range(1, k+1), metrics[metric], alpha=0.7)
        plt.axhline(y=avg_metrics[metric], color='r', linestyle='--', 
                   label=f'Mean: {avg_metrics[metric]:.4f}')
        plt.xlabel('Fold')
        plt.ylabel(metric.upper())
        plt.title(f'{metric.upper()} by Fold')
        plt.xticks(range(1, k+1))
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/evaluation/kfold_metrics.png', dpi=300)
    plt.close()
    
    return summary_df, avg_metrics, std_metrics
