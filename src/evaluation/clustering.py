"""
Embedding space clustering analysis module

This module provides functions to analyze the embedding space using clustering
techniques to assess the quality of the learned representations.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def analyze_embedding_space(embeddings, node_map, reverse_node_map, compound_list, target_list):
    """
    Analyze embedding space using clustering techniques
    
    Args:
        embeddings: Node embeddings
        node_map: Mapping from node IDs to indices
        reverse_node_map: Mapping from indices to node IDs
        compound_list: List of compound IDs
        target_list: List of target IDs
        
    Returns:
        Dictionary with clustering analysis results
    """
    print("Analyzing embedding space with clustering...")
    
    # Create output directory
    os.makedirs('results/evaluation/clustering', exist_ok=True)
    
    # Extract compound and target embeddings
    compound_indices = [node_map[comp_id] for comp_id in compound_list if comp_id in node_map]
    target_indices = [node_map[target_id] for target_id in target_list if target_id in node_map]
    
    compound_embeddings = embeddings[compound_indices].cpu().numpy()
    target_embeddings = embeddings[target_indices].cpu().numpy()
    
    # Prepare results
    results = {
        'compound_clustering': {},
        'target_clustering': {},
        'combined_clustering': {}
    }
    
    # Analyze compound embeddings
    results['compound_clustering'] = _cluster_analysis(
        compound_embeddings, 
        [reverse_node_map[idx] for idx in compound_indices],
        'Compound',
        prefix='compound'
    )
    
    # Analyze target embeddings
    results['target_clustering'] = _cluster_analysis(
        target_embeddings, 
        [reverse_node_map[idx] for idx in target_indices],
        'Target',
        prefix='target'
    )
    
    # Analyze combined embeddings
    combined_embeddings = embeddings.cpu().numpy()
    node_types = ['Compound' if i in compound_indices else 'Target' for i in range(len(embeddings))]
    
    results['combined_clustering'] = _cluster_analysis(
        combined_embeddings, 
        [reverse_node_map[i] for i in range(len(embeddings))],
        node_types,
        prefix='combined'
    )
    
    # Visualize embedding space using t-SNE
    _visualize_embeddings_tsne(
        embeddings.cpu().numpy(), 
        [reverse_node_map[i] for i in range(len(embeddings))],
        ['Compound' if i in compound_indices else 'Target' for i in range(len(embeddings))],
        'combined'
    )
    
    # Return results
    return results

def _cluster_analysis(embeddings, labels, node_types, prefix):
    """
    Perform cluster analysis on embeddings
    
    Args:
        embeddings: Embedding vectors
        labels: Node labels
        node_types: Node type labels
        prefix: Prefix for saving files
        
    Returns:
        Dictionary with clustering analysis results
    """
    results = {}
    
    # Try different numbers of clusters
    max_clusters = min(10, len(embeddings) // 2)
    silhouette_scores = []
    ch_scores = []
    
    # K-means clustering
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate clustering quality metrics
        try:
            silhouette = silhouette_score(embeddings, cluster_labels)
            ch_score = calinski_harabasz_score(embeddings, cluster_labels)
            
            silhouette_scores.append(silhouette)
            ch_scores.append(ch_score)
        except:
            silhouette_scores.append(0)
            ch_scores.append(0)
    
    # Find optimal number of clusters
    if silhouette_scores:
        optimal_n_clusters = np.argmax(silhouette_scores) + 2
        results['optimal_clusters'] = optimal_n_clusters
        results['silhouette_scores'] = silhouette_scores
        results['calinski_harabasz_scores'] = ch_scores
        
        # Perform clustering with optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Visualize clustering
        _visualize_embeddings_tsne(embeddings, labels, cluster_labels, f'{prefix}_clusters')
        
        # Create cluster summary
        cluster_data = []
        for i in range(optimal_n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            cluster_size = len(cluster_indices)
            
            if isinstance(node_types, list):
                # Count types in cluster
                type_counts = {}
                for idx in cluster_indices:
                    node_type = node_types[idx]
                    if node_type not in type_counts:
                        type_counts[node_type] = 0
                    type_counts[node_type] += 1
                
                cluster_data.append({
                    'cluster_id': i,
                    'size': cluster_size,
                    'silhouette': silhouette_score(embeddings, cluster_labels, metric='euclidean', sample_size=1000) if len(embeddings) > 1000 else silhouette_score(embeddings, cluster_labels),
                    **type_counts
                })
            else:
                cluster_data.append({
                    'cluster_id': i,
                    'size': cluster_size,
                    'silhouette': silhouette_score(embeddings, cluster_labels, metric='euclidean', sample_size=1000) if len(embeddings) > 1000 else silhouette_score(embeddings, cluster_labels),
                    node_types: cluster_size
                })
        
        # Save cluster summary
        cluster_df = pd.DataFrame(cluster_data)
        cluster_df.to_csv(f'results/evaluation/clustering/{prefix}_cluster_summary.csv', index=False)
        results['cluster_summary'] = cluster_df
        
        # Analyze within-cluster similarity
        within_cluster_sim = []
        between_cluster_sim = []
        
        # Sample for large datasets to avoid memory issues
        max_samples = 1000
        if len(embeddings) > max_samples:
            sample_indices = np.random.choice(len(embeddings), max_samples, replace=False)
            sample_embeddings = embeddings[sample_indices]
            sample_labels = cluster_labels[sample_indices]
        else:
            sample_embeddings = embeddings
            sample_labels = cluster_labels
        
        # Calculate cosine similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(sample_embeddings)
        
        # Analyze similarities
        for i in range(len(sample_embeddings)):
            for j in range(i+1, len(sample_embeddings)):
                sim = similarity_matrix[i, j]
                if sample_labels[i] == sample_labels[j]:
                    within_cluster_sim.append(sim)
                else:
                    between_cluster_sim.append(sim)
        
        # Calculate statistics
        within_mean = np.mean(within_cluster_sim)
        within_std = np.std(within_cluster_sim)
        between_mean = np.mean(between_cluster_sim)
        between_std = np.std(between_cluster_sim)
        
        results['within_cluster_similarity'] = {
            'mean': within_mean,
            'std': within_std
        }
        results['between_cluster_similarity'] = {
            'mean': between_mean,
            'std': between_std
        }
        
        # Plot similarity distributions
        plt.figure(figsize=(10, 6))
        plt.hist(within_cluster_sim, bins=30, alpha=0.7, label=f'Within Cluster (μ={within_mean:.3f})')
        plt.hist(between_cluster_sim, bins=30, alpha=0.7, label=f'Between Clusters (μ={between_mean:.3f})')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Count')
        plt.title(f'{prefix.title()} Embedding Similarity Distribution')
        plt.legend()
        plt.savefig(f'results/evaluation/clustering/{prefix}_similarity_distribution.png', dpi=300)
        plt.close()
        
        # Plot clustering quality metrics
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(2, max_clusters + 1), silhouette_scores, 'o-')
        plt.axvline(x=optimal_n_clusters, color='r', linestyle='--', 
                   label=f'Optimal: {optimal_n_clusters}')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score by Cluster Count')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(range(2, max_clusters + 1), ch_scores, 'o-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Calinski-Harabasz Score')
        plt.title('Calinski-Harabasz Score by Cluster Count')
        
        plt.tight_layout()
        plt.savefig(f'results/evaluation/clustering/{prefix}_cluster_quality.png', dpi=300)
        plt.close()
    
    return results

def _visualize_embeddings_tsne(embeddings, labels, color_labels, prefix):
    """
    Visualize embeddings using t-SNE
    
    Args:
        embeddings: Embedding vectors
        labels: Node labels
        color_labels: Labels for coloring points
        prefix: Prefix for saving files
    """
    # Apply t-SNE for visualization
    if len(embeddings) > 5000:
        # Sample for very large datasets
        sample_indices = np.random.choice(len(embeddings), 5000, replace=False)
        sample_embeddings = embeddings[sample_indices]
        
        if isinstance(color_labels, list):
            sample_colors = [color_labels[i] for i in sample_indices]
        else:
            sample_colors = color_labels[sample_indices]
            
        sample_labels = [labels[i] for i in sample_indices]
    else:
        sample_embeddings = embeddings
        sample_colors = color_labels
        sample_labels = labels
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sample_embeddings)-1))
    reduced_embeddings = tsne.fit_transform(sample_embeddings)
    
    # Plot embeddings
    plt.figure(figsize=(12, 10))
    
    # Color by label type
    if isinstance(sample_colors, list) and isinstance(sample_colors[0], str):
        # Get unique labels for coloring
        unique_labels = sorted(set(sample_colors))
        
        for label in unique_labels:
            indices = [i for i, l in enumerate(sample_colors) if l == label]
            if indices:
                plt.scatter(
                    reduced_embeddings[indices, 0], 
                    reduced_embeddings[indices, 1],
                    label=label,
                    alpha=0.7
                )
    else:
        # Numeric labels - use colormap
        scatter = plt.scatter(
            reduced_embeddings[:, 0], 
            reduced_embeddings[:, 1],
            c=sample_colors,
            cmap='viridis',
            alpha=0.7
        )
        plt.colorbar(scatter, label='Cluster')
    
    plt.title(f't-SNE Visualization of {prefix.title()} Embeddings')
    plt.tight_layout()
    
    if isinstance(sample_colors, list) and isinstance(sample_colors[0], str):
        plt.legend()
        
    plt.savefig(f'results/evaluation/clustering/{prefix}_tsne.png', dpi=300)
    plt.close()
