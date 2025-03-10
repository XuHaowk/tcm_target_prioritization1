"""
Semantic relationship analysis script

This script analyzes semantic relationships in the TCM interaction graph and 
calculates semantic relationship-based similarities.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.semantic_relation import RelationType, SemanticRelationHandler
from src.data.graph_builder import build_semantic_graph

def extract_relation_paths(data, node_map, reverse_node_map, relation_types, max_length=3):
    """
    Extract all semantic relation paths up to a maximum length
    
    Args:
        data: Graph data object
        node_map: Mapping from node IDs to indices
        reverse_node_map: Mapping from indices to node IDs
        relation_types: Dictionary mapping edge indices to relation types
        max_length: Maximum path length
        
    Returns:
        Dictionary of paths between compounds and targets
    """
    print(f"Extracting semantic relation paths (max length: {max_length})...")
    
    # Extract node indices by type
    compound_indices = data.compound_indices.tolist()
    target_indices = data.target_indices.tolist()
    
    # Extract edges
    edge_index = data.edge_index.cpu().numpy()
    edge_type = data.edge_type.cpu().numpy()
    
    # Create adjacency list for efficient path finding
    adj_list = {}
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        rel = relation_types[i]
        
        if src not in adj_list:
            adj_list[src] = []
        adj_list[src].append((dst, rel))
    
    # Extract paths using BFS
    paths = {}
    
    for compound_idx in tqdm(compound_indices, desc="Finding paths"):
        compound_id = reverse_node_map[compound_idx]
        
        for target_idx in target_indices:
            target_id = reverse_node_map[target_idx]
            
            # Find paths using BFS
            queue = [(compound_idx, [(compound_idx, None)])]  # (node, path)
            found_paths = []
            
            while queue:
                node, path = queue.pop(0)
                
                # If we have a path to target
                if node == target_idx and len(path) > 1:
                    # Extract relations from path
                    path_relations = []
                    for i in range(1, len(path)):
                        prev_node = path[i-1][0]
                        curr_node = path[i][0]
                        
                        # Find the relation between these nodes
                        for edge_idx in range(edge_index.shape[1]):
                            if edge_index[0, edge_idx] == prev_node and edge_index[1, edge_idx] == curr_node:
                                path_relations.append(relation_types[edge_idx])
                                break
                    
                    found_paths.append(path_relations)
                    continue
                
                # If path is already at max length, don't extend
                if len(path) > max_length:
                    continue
                
                # Extend path
                if node in adj_list:
                    for neighbor, rel in adj_list[node]:
                        # Avoid cycles
                        if neighbor not in [p[0] for p in path]:
                            queue.append((neighbor, path + [(neighbor, rel)]))
            
            if found_paths:
                key = (compound_id, target_id)
                paths[key] = found_paths
    
    return paths

def calculate_semantic_similarities(data, node_map, reverse_node_map, relation_types):
    """
    Calculate semantic relationship-based similarities
    
    Args:
        data: Graph data object
        node_map: Mapping from node IDs to indices
        reverse_node_map: Mapping from indices to node IDs
        relation_types: Dictionary mapping edge indices to relation types
        
    Returns:
        DataFrame with semantic similarities
    """
    print("Calculating semantic relationship similarities...")
    
    # Extract node indices by type
    compound_indices = data.compound_indices.tolist()
    target_indices = data.target_indices.tolist()
    
    # Extract relation paths
    paths = extract_relation_paths(data, node_map, reverse_node_map, relation_types)
    
    # Initialize semantic relation handler
    relation_handler = SemanticRelationHandler()
    
    # Calculate semantic similarities
    results = []
    
    for (compound_id, target_id), compound_target_paths in tqdm(paths.items(), desc="Calculating similarities"):
        # Calculate path-driven semantic similarity
        max_path_similarity = 0.0
        
        for path_relations in compound_target_paths:
            path_similarity = relation_handler.calculate_path_similarity(path_relations)
            max_path_similarity = max(max_path_similarity, path_similarity)
        
        # Create result entry
        results.append({
            'compound_id': compound_id,
            'target_id': target_id,
            'semantic_similarity': max_path_similarity,
            'num_paths': len(compound_target_paths)
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    os.makedirs('results/semantic_analysis', exist_ok=True)
    df.to_csv('results/semantic_analysis/semantic_similarities.csv', index=False)
    print(f"Semantic similarities saved to results/semantic_analysis/semantic_similarities.csv")
    
    return df

def analyze_meta_paths(data, node_map, reverse_node_map, relation_types):
    """
    Analyze meta-paths in the graph
    
    Args:
        data: Graph data object
        node_map: Mapping from node IDs to indices
        reverse_node_map: Mapping from indices to node IDs
        relation_types: Dictionary mapping edge indices to relation types
        
    Returns:
        DataFrame with meta-path analysis
    """
    print("Analyzing meta-paths...")
    
    # Extract relation paths
    paths = extract_relation_paths(data, node_map, reverse_node_map, relation_types)
    
    # Count meta-paths
    meta_path_counts = {}
    
    for (compound_id, target_id), compound_target_paths in tqdm(paths.items(), desc="Counting meta-paths"):
        for path_relations in compound_target_paths:
            # Create meta-path signature
            meta_path = tuple(rel.value for rel in path_relations)
            
            if meta_path not in meta_path_counts:
                meta_path_counts[meta_path] = 0
            meta_path_counts[meta_path] += 1
    
    # Create DataFrame
    meta_path_data = []
    for meta_path, count in sorted(meta_path_counts.items(), key=lambda x: x[1], reverse=True):
        meta_path_data.append({
            'meta_path': '->'.join(meta_path),
            'count': count,
            'length': len(meta_path)
        })
    
    df = pd.DataFrame(meta_path_data)
    
    # Save results
    df.to_csv('results/semantic_analysis/meta_paths.csv', index=False)
    print(f"Meta-path analysis saved to results/semantic_analysis/meta_paths.csv")
    
    return df

def create_semantic_similarity_matrix(df, compound_list, target_list):
    """
    Create semantic similarity matrix from semantic similarity results
    
    Args:
        df: DataFrame with semantic similarities
        compound_list: List of compound IDs
        target_list: List of target IDs
        
    Returns:
        Semantic similarity matrix
    """
    # Create similarity matrix
    similarity_matrix = np.zeros((len(compound_list), len(target_list)))
    
    # Create maps from IDs to indices
    compound_to_idx = {comp_id: idx for idx, comp_id in enumerate(compound_list)}
    target_to_idx = {target_id: idx for idx, target_id in enumerate(target_list)}
    
    # Fill similarity matrix
    for _, row in df.iterrows():
        compound_id = row['compound_id']
        target_id = row['target_id']
        
        if compound_id in compound_to_idx and target_id in target_to_idx:
            comp_idx = compound_to_idx[compound_id]
            target_idx = target_to_idx[target_id]
            similarity_matrix[comp_idx, target_idx] = row['semantic_similarity']
    
    return similarity_matrix

def main():
    """Main function"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Semantic Relationship Analysis')
    parser.add_argument('--kg_data', type=str, default='data/raw/kg_data_extended.csv',
                       help='Knowledge graph data file')
    parser.add_argument('--db_data', type=str, default='data/raw/database_data_extended.csv',
                       help='Database data file')
    parser.add_argument('--output_dir', type=str, default='results/semantic_analysis',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load knowledge graph data
    kg_data = pd.read_csv(args.kg_data)
    print(f"Loaded knowledge graph with {len(kg_data)} relations")
    
    # Load database data to get compound features
    compound_features = {}
    if os.path.exists(args.db_data):
        db_data = pd.read_csv(args.db_data)
        print(f"Loaded database with {len(db_data)} compounds")
        
        # Extract compound features
        for _, row in db_data.iterrows():
            comp_id = row.get('compound_id', row.get('compound', f"compound_{_}"))
            try:
                # Parse feature vector string to list of floats
                if 'feature_vector' in row:
                    features = eval(row['feature_vector'])
                    compound_features[comp_id] = torch.tensor(features, dtype=torch.float32)
                else:
                    # Create random features if no feature vector
                    compound_features[comp_id] = torch.randn(256, dtype=torch.float32)
            except Exception as e:
                print(f"Error parsing features for {comp_id}: {e}")
                # Create random features as fallback
                compound_features[comp_id] = torch.randn(256, dtype=torch.float32)
    else:
        print(f"Warning: Database file not found: {args.db_data}")
        print("Creating empty compound features dictionary")
    
    # Build semantic graph
    data, node_map, reverse_node_map, relation_types = build_semantic_graph(
        kg_data, None, compound_features, None, None, None)
    
    # Calculate semantic similarities
    semantic_similarities = calculate_semantic_similarities(data, node_map, reverse_node_map, relation_types)
    
    # Analyze meta-paths
    meta_paths = analyze_meta_paths(data, node_map, reverse_node_map, relation_types)
    
    # Print summary
    print("\n=== Semantic Analysis Summary ===")
    print(f"Total semantic relationships analyzed: {len(semantic_similarities)}")
    print(f"Number of unique meta-paths identified: {len(meta_paths)}")
    if not meta_paths.empty:
        print(f"Most common meta-path: {meta_paths.iloc[0]['meta_path']} (count: {meta_paths.iloc[0]['count']})")
    
    print("\nSemantic analysis completed!")

if __name__ == "__main__":
    main()
