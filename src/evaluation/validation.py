"""
Validation module for TCM target prioritization

This module provides functions to validate model predictions against known interactions
and calculate evaluation metrics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score

def create_validation_matrix(validated_data, compound_list, target_list):
    """
    Create a binary interaction matrix from validated data
    
    Args:
        validated_data: DataFrame with validated interactions
        compound_list: List of compound IDs
        target_list: List of target IDs
        
    Returns:
        Binary matrix of validated interactions
    """
    # Create maps from IDs to indices
    compound_to_idx = {comp_id: idx for idx, comp_id in enumerate(compound_list)}
    target_to_idx = {target_id: idx for idx, target_id in enumerate(target_list)}
    
    # Create validation matrix
    validation_matrix = np.zeros((len(compound_list), len(target_list)))
    
    # Fill in known interactions
    interaction_count = 0
    
    # Check column names in the DataFrame first
    compound_col = 'compound' if 'compound' in validated_data.columns else 'compound_id'
    target_col = 'target' if 'target' in validated_data.columns else 'target_id'
    
    for _, row in validated_data.iterrows():
        # Use .get() to safely handle missing values
        comp_id = row.get(compound_col, '')
        target_id = row.get(target_col, '')
        
        if comp_id in compound_to_idx and target_id in target_to_idx:
            comp_idx = compound_to_idx[comp_id]
            target_idx = target_to_idx[target_id]
            validation_matrix[comp_idx, target_idx] = 1
            interaction_count += 1
    
    print(f"Created validation matrix with {interaction_count} known interactions")
    
    return validation_matrix

def create_prediction_matrix(priorities, compound_list, target_list):
    """
    Create a prediction score matrix from prioritized targets
    
    Args:
        priorities: Dictionary mapping compounds to prioritized targets
        compound_list: List of compound IDs
        target_list: List of target IDs
        
    Returns:
        Matrix of prediction scores
    """
    # Create maps from IDs to indices
    compound_to_idx = {comp_id: idx for idx, comp_id in enumerate(compound_list)}
    target_to_idx = {target_id: idx for idx, target_id in enumerate(target_list)}
    
    # Create prediction matrix
    prediction_matrix = np.zeros((len(compound_list), len(target_list)))
    
    # Fill with prediction scores
    for comp_id, target_scores in priorities.items():
        if comp_id in compound_to_idx:
            comp_idx = compound_to_idx[comp_id]
            for target_id, score in target_scores.items():
                if target_id in target_to_idx:
                    target_idx = target_to_idx[target_id]
                    prediction_matrix[comp_idx, target_idx] = score
    
    return prediction_matrix

def calculate_metrics(validation_matrix, prediction_matrix):
    """
    Calculate evaluation metrics
    
    Args:
        validation_matrix: Binary matrix of validated interactions
        prediction_matrix: Matrix of prediction scores
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Flatten matrices for evaluation
    y_true = validation_matrix.flatten()
    y_pred = prediction_matrix.flatten()
    
    # Calculate precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred)
    
    # Calculate average precision
    average_precision = average_precision_score(y_true, y_pred)
    
    # Calculate ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Calculate hit rates
    hit_rates = {}
    for k in [1, 3, 5, 10, 20]:
        hit_count = 0
        total_count = 0
        
        for i in range(validation_matrix.shape[0]):
            # Get validated targets for this compound
            valid_targets = np.where(validation_matrix[i] == 1)[0]
            if len(valid_targets) > 0:
                total_count += len(valid_targets)
                
                # Get top k predicted targets
                pred_scores = prediction_matrix[i]
                top_indices = np.argsort(pred_scores)[-k:]
                
                # Count hits
                for target_idx in valid_targets:
                    if target_idx in top_indices:
                        hit_count += 1
        
        hit_rates[f'hit@{k}'] = hit_count / max(1, total_count)
    
    # Calculate Mean Reciprocal Rank
    mrr_values = []
    for i in range(validation_matrix.shape[0]):
        valid_targets = np.where(validation_matrix[i] == 1)[0]
        if len(valid_targets) > 0:
            pred_scores = prediction_matrix[i]
            # Get ranks (higher score = lower rank number)
            ranks = len(pred_scores) - np.argsort(np.argsort(pred_scores))[valid_targets]
            # Calculate reciprocal ranks
            mrr_values.extend(1.0 / ranks)
    
    mrr = np.mean(mrr_values) if mrr_values else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'pr_thresholds': pr_thresholds,
        'average_precision': average_precision,
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': roc_thresholds,
        'roc_auc': roc_auc,
        'hit_rates': hit_rates,
        'mrr': mrr
    }

def plot_validation_metrics(metrics, output_file=None):
    """
    Create plots of validation metrics
    
    Args:
        metrics: Dictionary of evaluation metrics
        output_file: Path to save the visualization
    """
    plt.figure(figsize=(18, 6))
    
    # Plot precision-recall curve
    plt.subplot(1, 3, 1)
    plt.step(metrics['recall'], metrics['precision'], color='b', alpha=0.8, where='post')
    plt.fill_between(metrics['recall'], metrics['precision'], alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AP={metrics["average_precision"]:.3f})')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot ROC curve
    plt.subplot(1, 3, 2)
    plt.plot(metrics['fpr'], metrics['tpr'], color='r', alpha=0.8)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC={metrics["roc_auc"]:.3f})')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot hit rates
    plt.subplot(1, 3, 3)
    hit_rates = metrics['hit_rates']
    k_values = sorted([int(k.split('@')[1]) for k in hit_rates.keys()])
    rates = [hit_rates[f'hit@{k}'] for k in k_values]
    
    plt.bar(range(len(k_values)), rates, tick_label=[f'Hit@{k}' for k in k_values])
    plt.xlabel('Metric')
    plt.ylabel('Rate')
    plt.title('Hit Rates at Different K Values')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add hit rate values above the bars
    for i, v in enumerate(rates):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    
    # Save figure if output file provided
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        plt.savefig(output_file, dpi=300)
        print(f"Validation metrics plot saved to {output_file}")
    
    plt.close()

def validate_model(priorities, validated_data, compound_list, target_list, output_file=None):
    """
    Validate model predictions against known data
    
    Args:
        priorities: Dictionary mapping compounds to prioritized targets
        validated_data: DataFrame with validated interactions
        compound_list: List of compound IDs
        target_list: List of target IDs
        output_file: Path to save visualization
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Create validation matrix
    validation_matrix = create_validation_matrix(validated_data, compound_list, target_list)
    
    # Create prediction matrix
    prediction_matrix = create_prediction_matrix(priorities, compound_list, target_list)
    
    # Calculate metrics
    metrics = calculate_metrics(validation_matrix, prediction_matrix)
    
    # Create visualization
    plot_validation_metrics(metrics, output_file)
    
    return metrics
