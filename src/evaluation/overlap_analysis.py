"""
Target overlap analysis module

This module provides functions to analyze the stability and overlap of 
top predicted targets across different runs or configurations.
"""
import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import kendalltau

def calculate_top_n_overlap(predictions_list, compound_list, top_n_values=None):
    """
    Calculate overlap of top N predicted targets across multiple runs
    
    Args:
        predictions_list: List of prediction dictionaries from different runs
        compound_list: List of compound IDs
        top_n_values: List of N values to calculate overlap for
        
    Returns:
        Dictionary with overlap analysis results
    """
    if top_n_values is None:
        top_n_values = [5, 10, 20]
    
    print(f"Calculating target overlap for top {top_n_values} predictions...")
    
    results = {
        'jaccard': {N: [] for N in top_n_values},
        'overlap_count': {N: [] for N in top_n_values},
        'kendall_tau': []
    }
    
    # Calculate pairwise overlaps
    num_runs = len(predictions_list)
    
    for i in range(num_runs):
        for j in range(i+1, num_runs):
            run_i = predictions_list[i]
            run_j = predictions_list[j]
            
            compound_overlaps = {N: [] for N in top_n_values}
            compound_kendall = []
            
            for comp_id in compound_list:
                if comp_id in run_i and comp_id in run_j:
                    # Extract top targets from each run
                    targets_i = run_i[comp_id]
                    targets_j = run_j[comp_id]
                    
                    # Calculate Kendall's Tau for common targets
                    common_targets = set(targets_i.keys()) & set(targets_j.keys())
                    if len(common_targets) > 1:
                        # Extract ranks
                        rank_i = {t: list(targets_i.keys()).index(t) for t in common_targets}
                        rank_j = {t: list(targets_j.keys()).index(t) for t in common_targets}
                        
                        # Sort by target ID for consistent ordering
                        targets_sorted = sorted(common_targets)
                        ranks_i = [rank_i[t] for t in targets_sorted]
                        ranks_j = [rank_j[t] for t in targets_sorted]
                        
                        # Calculate Kendall's Tau
                        tau, _ = kendalltau(ranks_i, ranks_j)
                        if not np.isnan(tau):
                            compound_kendall.append(tau)
                    
                    # Calculate overlaps for different N values
                    for N in top_n_values:
                        top_n_i = list(targets_i.keys())[:N]
                        top_n_j = list(targets_j.keys())[:N]
                        
                        # Calculate Jaccard similarity
                        intersection = set(top_n_i) & set(top_n_j)
                        union = set(top_n_i) | set(top_n_j)
                        
                        if union:
                            jaccard = len(intersection) / len(union)
                            compound_overlaps[N].append(jaccard)
            
            # Store results
            for N in top_n_values:
                if compound_overlaps[N]:
                    avg_jaccard = np.mean(compound_overlaps[N])
                    results['jaccard'][N].append(avg_jaccard)
                    results['overlap_count'][N].append(len(compound_overlaps[N]))
            
            if compound_kendall:
                avg_kendall = np.mean(compound_kendall)
                results['kendall_tau'].append(avg_kendall)
    
    # Calculate overall statistics
    overall_results = {
        'jaccard_mean': {N: np.mean(results['jaccard'][N]) if results['jaccard'][N] else 0 
                        for N in top_n_values},
        'jaccard_std': {N: np.std(results['jaccard'][N]) if len(results['jaccard'][N]) > 1 else 0 
                       for N in top_n_values},
        'kendall_tau_mean': np.mean(results['kendall_tau']) if results['kendall_tau'] else 0,
        'kendall_tau_std': np.std(results['kendall_tau']) if len(results['kendall_tau']) > 1 else 0
    }
    
    # Create summary DataFrame
    summary_data = []
    for N in top_n_values:
        summary_data.append({
            'top_n': N,
            'mean_jaccard': overall_results['jaccard_mean'][N],
            'std_jaccard': overall_results['jaccard_std'][N],
            'total_comparisons': len(results['jaccard'][N])
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save results
    os.makedirs('results/evaluation/overlap', exist_ok=True)
    summary_df.to_csv('results/evaluation/overlap/target_overlap_summary.csv', index=False)
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.bar([str(N) for N in top_n_values], 
            [overall_results['jaccard_mean'][N] for N in top_n_values],
            yerr=[overall_results['jaccard_std'][N] for N in top_n_values],
            alpha=0.7)
    plt.xlabel('Top N')
    plt.ylabel('Mean Jaccard Similarity')
    plt.title('Target Overlap Analysis')
    plt.ylim(0, 1)
    
    # Add value labels
    for i, N in enumerate(top_n_values):
        plt.text(i, overall_results['jaccard_mean'][N] + 0.02, 
                f"{overall_results['jaccard_mean'][N]:.3f}", 
                ha='center')
    
    plt.savefig('results/evaluation/overlap/target_overlap.png', dpi=300)
    plt.close()
    
    # Print results
    print("\nTarget Overlap Analysis Results:")
    for N in top_n_values:
        print(f"  Top {N}: Jaccard = {overall_results['jaccard_mean'][N]:.3f} ± {overall_results['jaccard_std'][N]:.3f}")
    
    if results['kendall_tau']:
        print(f"  Kendall's Tau = {overall_results['kendall_tau_mean']:.3f} ± {overall_results['kendall_tau_std']:.3f}")
    
    return overall_results, summary_df

def analyze_stability(predictions_list, compound_list, target_list):
    """
    Analyze stability of predictions across multiple runs
    
    Args:
        predictions_list: List of prediction dictionaries from different runs
        compound_list: List of compound IDs
        target_list: List of target IDs
        
    Returns:
        Dictionary with stability analysis results
    """
    print("Analyzing prediction stability...")
    
    # Create score matrices
    num_runs = len(predictions_list)
    score_matrices = []
    
    for run_idx, predictions in enumerate(predictions_list):
        # Create prediction matrix
        matrix = np.zeros((len(compound_list), len(target_list)))
        
        # Create maps from IDs to indices
        compound_to_idx = {comp_id: idx for idx, comp_id in enumerate(compound_list)}
        target_to_idx = {target_id: idx for idx, target_id in enumerate(target_list)}
        
        # Fill prediction matrix
        for comp_id, target_scores in predictions.items():
            if comp_id in compound_to_idx:
                comp_idx = compound_to_idx[comp_id]
                for target_id, score in target_scores.items():
                    if target_id in target_to_idx:
                        target_idx = target_to_idx[target_id]
                        matrix[comp_idx, target_idx] = score
        
        score_matrices.append(matrix)
    
    # Calculate coefficient of variation
    if num_runs >= 2:
        # Stack matrices
        stacked = np.stack(score_matrices, axis=0)
        
        # Calculate mean and standard deviation across runs
        mean_scores = np.mean(stacked, axis=0)
        std_scores = np.std(stacked, axis=0)
        
        # Calculate coefficient of variation (CV)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        cv = std_scores / (mean_scores + epsilon)
        
        # Calculate statistics
        mean_cv = np.mean(cv)
        median_cv = np.median(cv)
        max_cv = np.max(cv)
        
        # Identify most and least stable predictions
        flat_cv = cv.flatten()
        flat_mean = mean_scores.flatten()
        
        # Filter out zero means
        valid_indices = np.where(flat_mean > epsilon)[0]
        valid_cv = flat_cv[valid_indices]
        
        if len(valid_cv) > 0:
            # Most stable (lowest CV)
            most_stable_idx = valid_indices[np.argmin(valid_cv)]
            comp_idx, target_idx = np.unravel_index(most_stable_idx, cv.shape)
            most_stable = {
                'compound_id': compound_list[comp_idx],
                'target_id': target_list[target_idx],
                'cv': flat_cv[most_stable_idx],
                'mean_score': flat_mean[most_stable_idx]
            }
            
            # Least stable (highest CV)
            least_stable_idx = valid_indices[np.argmax(valid_cv)]
            comp_idx, target_idx = np.unravel_index(least_stable_idx, cv.shape)
            least_stable = {
                'compound_id': compound_list[comp_idx],
                'target_id': target_list[target_idx],
                'cv': flat_cv[least_stable_idx],
                'mean_score': flat_mean[least_stable_idx]
            }
        else:
            most_stable = least_stable = {
                'compound_id': None,
                'target_id': None,
                'cv': None,
                'mean_score': None
            }
        
        # Calculate per-compound stability
        compound_stability = []
        for i, comp_id in enumerate(compound_list):
            comp_cv = cv[i, :]
            comp_mean = mean_scores[i, :]
            
            # Filter out zero means
            valid_indices = np.where(comp_mean > epsilon)[0]
            if len(valid_indices) > 0:
                valid_cv = comp_cv[valid_indices]
                mean_cv = np.mean(valid_cv)
                
                compound_stability.append({
                    'compound_id': comp_id,
                    'mean_cv': mean_cv,
                    'target_count': len(valid_indices)
                })
        
        # Sort compounds by stability
        compound_stability.sort(key=lambda x: x['mean_cv'])
        
        # Save results
        compound_stability_df = pd.DataFrame(compound_stability)
        compound_stability_df.to_csv('results/evaluation/overlap/compound_stability.csv', index=False)
        
        # Visualize CV distribution
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(cv.flatten(), bins=30, alpha=0.7)
        plt.axvline(x=mean_cv, color='r', linestyle='--', label=f'Mean: {mean_cv:.3f}')
        plt.axvline(x=median_cv, color='g', linestyle='--', label=f'Median: {median_cv:.3f}')
        plt.xlabel('Coefficient of Variation')
        plt.ylabel('Count')
        plt.title('Distribution of Prediction Stability')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        if len(compound_stability) > 0:
            # Plot top and bottom 10 compounds by stability
            top_n = min(10, len(compound_stability))
            top_compounds = compound_stability[:top_n]
            bottom_compounds = compound_stability[-top_n:]
            
            # Combine and sort for plotting
            plot_data = top_compounds + bottom_compounds
            plot_data.sort(key=lambda x: x['mean_cv'])
            
            plt.barh([x['compound_id'] for x in plot_data], 
                    [x['mean_cv'] for x in plot_data], 
                    alpha=0.7)
            plt.xlabel('Mean Coefficient of Variation')
            plt.ylabel('Compound')
            plt.title('Most and Least Stable Compounds')
        
        plt.tight_layout()
        plt.savefig('results/evaluation/overlap/prediction_stability.png', dpi=300)
        plt.close()
        
        # Return results
        stability_results = {
            'mean_cv': mean_cv,
            'median_cv': median_cv,
            'max_cv': max_cv,
            'most_stable': most_stable,
            'least_stable': least_stable,
            'compound_stability': compound_stability
        }
        
        # Print summary
        print("\nPrediction Stability Analysis:")
        print(f"  Mean CV: {mean_cv:.3f}")
        print(f"  Median CV: {median_cv:.3f}")
        print(f"  Most stable prediction: {most_stable['compound_id']} - {most_stable['target_id']} (CV = {most_stable['cv']:.3f})")
        print(f"  Least stable prediction: {least_stable['compound_id']} - {least_stable['target_id']} (CV = {least_stable['cv']:.3f})")
        
        return stability_results
    else:
        print("Need at least 2 runs for stability analysis")
        return None
