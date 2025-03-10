"""
Evaluation metrics module

This module provides functions for calculating and plotting evaluation metrics
for the TCM target prioritization system.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, roc_auc_score


def calculate_metrics(predictions, ground_truth, k_values=[1, 3, 5, 10]):
    """
    Calculate evaluation metrics for target prioritization
    
    Args:
        predictions: DataFrame with target predictions and scores
        ground_truth: DataFrame with validated interactions
        k_values: List of k values for Hit@k metrics
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Extract compound-target pairs from predictions
    predicted_pairs = {}
    compound_targets = {}
    
    for _, row in predictions.iterrows():
        compound_id = row['compound_id']
        target_id = row['target_id']
        score = row['final_score']
        
        if compound_id not in predicted_pairs:
            predicted_pairs[compound_id] = []
            compound_targets[compound_id] = []
        
        predicted_pairs[compound_id].append((target_id, score))
        compound_targets[compound_id].append(target_id)
    
    # Sort predictions by score for each compound
    for compound_id in predicted_pairs:
        predicted_pairs[compound_id].sort(key=lambda x: x[1], reverse=True)
    
    # Extract validated compound-target pairs
    validated_pairs = set()
    validated_compounds = set()
    validated_targets = set()
    
    for _, row in ground_truth.iterrows():
        compound_id = row['compound_id'] if 'compound_id' in row else row.get('compound', '')
        target_id = row['target_id'] if 'target_id' in row else row.get('target', '')
        
        if compound_id and target_id:
            validated_pairs.add((compound_id, target_id))
            validated_compounds.add(compound_id)
            validated_targets.add(target_id)
    
    # Calculate Hit@k
    hits_at_k = {k: 0 for k in k_values}
    total_compounds = 0
    
    for compound_id, target_scores in predicted_pairs.items():
        # Skip compounds not in validated data
        if compound_id not in validated_compounds:
            continue
        
        total_compounds += 1
        predictions_at_k = {}
        
        # Get top k predictions for each k
        for k in k_values:
            predictions_at_k[k] = [target for target, _ in target_scores[:k]]
        
        # Check if any validated target is in the top k predictions
        for target_id in validated_targets:
            if (compound_id, target_id) in validated_pairs:
                for k in k_values:
                    if target_id in predictions_at_k[k]:
                        hits_at_k[k] += 1
                        break
    
    # Calculate Hit@k ratios
    hit_at_k_ratios = {f"hit@{k}": hits_at_k[k] / total_compounds if total_compounds > 0 else 0.0 
                       for k in k_values}
    
    # Calculate Mean Reciprocal Rank (MRR)
    mrr = 0.0
    
    for compound_id, target_scores in predicted_pairs.items():
        # Skip compounds not in validated data
        if compound_id not in validated_compounds:
            continue
        
        # Get rank of first validated target
        for i, (target_id, _) in enumerate(target_scores):
            if (compound_id, target_id) in validated_pairs:
                mrr += 1.0 / (i + 1)
                break
    
    # Calculate average MRR
    mrr /= total_compounds if total_compounds > 0 else 1.0
    
    # Calculate precision-recall curve and AUC
    y_true = []
    y_scores = []
    
    for compound_id, target_scores in predicted_pairs.items():
        for target_id, score in target_scores:
            is_validated = 1 if (compound_id, target_id) in validated_pairs else 0
            y_true.append(is_validated)
            y_scores.append(score)
    
    # Skip if no predictions or all true/false
    if not y_true or len(set(y_true)) != 2:
        pr_auc = np.nan
        roc_auc = np.nan
        precision = []
        recall = []
        fpr = []
        tpr = []
    else:
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
    
    # Compile all metrics
    metrics = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "mrr": mrr,
        **hit_at_k_ratios,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "tpr": tpr
    }
    
    return metrics


def plot_validation_metrics(metrics, save_path=None):
    """
    Plot validation metrics
    
    Args:
        metrics: Dictionary with evaluation metrics
        save_path: Path to save the plot
        
    Returns:
        Figure object
    """
    # Create figure with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Set figure title
    fig.suptitle('Validation Metrics', fontsize=16)
    
    # Plot precision-recall curve
    if isinstance(metrics.get("precision", []), np.ndarray) and len(metrics.get("precision", [])) > 0:
        axes[0, 0].plot(metrics["recall"], metrics["precision"], label=f'PR AUC = {metrics["pr_auc"]:.3f}')
        axes[0, 0].set_title('Precision-Recall Curve')
        axes[0, 0].set_xlabel('Recall')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    else:
        axes[0, 0].text(0.5, 0.5, 'Insufficient data for PR curve', 
                        horizontalalignment='center', verticalalignment='center')
    
    # Plot ROC curve
    if isinstance(metrics.get("fpr", []), np.ndarray) and len(metrics.get("fpr", [])) > 0:
        axes[0, 1].plot(metrics["fpr"], metrics["tpr"], label=f'ROC AUC = {metrics["roc_auc"]:.3f}')
        axes[0, 1].plot([0, 1], [0, 1], 'k--')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    else:
        axes[0, 1].text(0.5, 0.5, 'Insufficient data for ROC curve', 
                        horizontalalignment='center', verticalalignment='center')
    
    # Plot Hit@k
    hit_k_metrics = {k: v for k, v in metrics.items() if k.startswith("hit@")}
    if hit_k_metrics:
        k_values = [int(k.split("@")[1]) for k in hit_k_metrics.keys()]
        hit_rates = list(hit_k_metrics.values())
        
        axes[1, 0].bar(k_values, hit_rates)
        axes[1, 0].set_title('Hit@k')
        axes[1, 0].set_xlabel('k')
        axes[1, 0].set_ylabel('Hit Rate')
        axes[1, 0].grid(True, axis='y')
        
        # Add value labels on top of bars
        for i, v in enumerate(hit_rates):
            axes[1, 0].text(k_values[i], v + 0.02, f'{v:.2f}', 
                            ha='center', va='bottom', fontweight='bold')
    else:
        axes[1, 0].text(0.5, 0.5, 'No Hit@k metrics available', 
                        horizontalalignment='center', verticalalignment='center')
    
    # Plot MRR
    if "mrr" in metrics:
        axes[1, 1].bar(['MRR'], [metrics["mrr"]])
        axes[1, 1].set_title('Mean Reciprocal Rank')
        axes[1, 1].set_ylabel('MRR')
        axes[1, 1].grid(True, axis='y')
        
        # Add value label on top of bar
        axes[1, 1].text(0, metrics["mrr"] + 0.02, f'{metrics["mrr"]:.3f}', 
                        ha='center', va='bottom', fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'No MRR metric available', 
                        horizontalalignment='center', verticalalignment='center')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Validation metrics plot saved to {save_path}")
    
    return fig
