"""
Enhanced graph builder module with semantic relation support

This module provides functions to build a heterogeneous graph from knowledge graph data,
incorporating semantic relationships and other similarities.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import json
import sys

# Add the project root to the path so Python can find the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from src.data.semantic_relation import RelationType, SemanticRelationHandler
except ImportError:
    # Define a simplified version here if import fails
    class RelationType:
        ACTIVATION = "activation"
        INHIBITION = "inhibition"
        BINDING = "binding"
        MODULATION = "modulation"
        SUBSTRATE = "substrate"
        TRANSPORT = "transport"
        INDIRECT_REGULATION = "indirect_regulation"
        UNKNOWN = "unknown"
        
        @classmethod
        def from_string(cls, relation_str):
            relation_str = relation_str.lower() if relation_str else ""
            
            if relation_str == "activation":
                return cls.ACTIVATION
            elif relation_str == "inhibition":
                return cls.INHIBITION
            elif relation_str == "binding":
                return cls.BINDING
            elif relation_str == "modulation":
                return cls.MODULATION
            elif relation_str == "substrate":
                return cls.SUBSTRATE
            elif relation_str == "transport":
                return cls.TRANSPORT
            elif relation_str == "indirect_regulation":
                return cls.INDIRECT_REGULATION
            else:
                return cls.UNKNOWN
    
    class SemanticRelationHandler:
        def __init__(self):
            self.relation_weights = {
                RelationType.ACTIVATION: 0.9,
                RelationType.INHIBITION: 0.9,
                RelationType.MODULATION: 0.7,
                RelationType.BINDING: 0.5,
                RelationType.SUBSTRATE: 0.7,
                RelationType.TRANSPORT: 0.6,
                RelationType.INDIRECT_REGULATION: 0.4,
                RelationType.UNKNOWN: 0.3
            }
        
        def get_relation_weight(self, relation_type):
            if isinstance(relation_type, str):
                relation_type = RelationType.from_string(relation_type)
            
            return self.relation_weights.get(relation_type, self.relation_weights[RelationType.UNKNOWN])

def build_semantic_graph(kg_data, external_data=None, compound_features=None, target_features=None,
                        drug_similarity=None, protein_similarity=None,
                        drug_sim_threshold=0.5, protein_sim_threshold=0.5):
    """
    Build heterogeneous graph with semantic relationship support
    
    Args:
        kg_data: Knowledge graph DataFrame
        external_data: External data for additional edges
        compound_features: Dictionary of compound features
        target_features: Dictionary of target features
        drug_similarity: DataFrame with drug similarity matrix
        protein_similarity: DataFrame with protein similarity matrix
        drug_sim_threshold: Threshold for creating similarity edges between drugs
        protein_sim_threshold: Threshold for creating similarity edges between proteins
        
    Returns:
        data: PyG Data object
        node_map: Mapping from node IDs to indices
        reverse_node_map: Mapping from indices to node IDs
        relation_types: Dictionary mapping edge indices to relation types
    """
    # Initialize features dictionaries if None
    if compound_features is None:
        compound_features = {}
    if target_features is None:
        target_features = {}
    
    # Extract all nodes from KG
    compounds = set()
    targets = set()
    
    for _, row in kg_data.iterrows():
        comp = row['compound'] if 'compound' in row else row.get('compound_id', '')
        target = row['target'] if 'target' in row else row.get('target_id', '')
        
        if comp:
            compounds.add(comp)
        if target:
            targets.add(target)
    
    # Create mapping from node names to indices
    all_nodes = list(compounds) + list(targets)
    node_map = {node: i for i, node in enumerate(all_nodes)}
    reverse_node_map = {i: node for node, i in node_map.items()}
    
    # Save mappings as JSON for later use
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/node_map.json', 'w') as f:
        json.dump({str(k): v for k, v in node_map.items()}, f)
    with open('data/processed/reverse_node_map.json', 'w') as f:
        json.dump({str(k): v for k, v in reverse_node_map.items()}, f)
    
    # Initialize semantic relation handler
    relation_handler = SemanticRelationHandler()
    
    # Prepare edge indices, weights and relation types from KG
    src_nodes = []
    dst_nodes = []
    edge_weights = []
    edge_types = []  # Store relation type for each edge
    edge_type_indices = []  # Numerical index for each relation type
    
    # Map relation types to indices
    relation_type_to_idx = {
        RelationType.ACTIVATION: 0,
        RelationType.INHIBITION: 1,
        RelationType.MODULATION: 2,
        RelationType.BINDING: 3,
        RelationType.SUBSTRATE: 4,
        RelationType.TRANSPORT: 5,
        RelationType.INDIRECT_REGULATION: 6,
        RelationType.UNKNOWN: 7
    }
    
    for _, row in kg_data.iterrows():
        comp = row['compound'] if 'compound' in row else row.get('compound_id', '')
        target = row['target'] if 'target' in row else row.get('target_id', '')
        confidence = row.get('confidence_score', 1.0)
        relation_type_str = row.get('relation_type', 'unknown')
        
        if comp in node_map and target in node_map:
            # Get relation type
            relation_type = RelationType.from_string(relation_type_str)
            relation_weight = relation_handler.get_relation_weight(relation_type)
            
            # Add edge from compound to target
            src_nodes.append(node_map[comp])
            dst_nodes.append(node_map[target])
            # Weight by confidence and relation importance
            edge_weights.append(float(confidence) * relation_weight)
            edge_types.append(relation_type)
            edge_type_indices.append(relation_type_to_idx.get(relation_type, 7))  # Default to UNKNOWN (7)
            
            # Add reverse edge (for undirected graph)
            src_nodes.append(node_map[target])
            dst_nodes.append(node_map[comp])
            edge_weights.append(float(confidence) * relation_weight)
            edge_types.append(relation_type)
            edge_type_indices.append(relation_type_to_idx.get(relation_type, 7))
    
    # Add drug similarity edges if provided
    if drug_similarity is not None:
        print("Adding drug similarity edges...")
        drug_ids = drug_similarity.columns
        
        for i, drug_i in enumerate(drug_ids):
            if drug_i not in node_map:
                continue
                
            for j, drug_j in enumerate(drug_ids):
                if i != j and drug_j in node_map:
                    sim_score = drug_similarity.iloc[i, j]
                    
                    # Only add edges for similar drugs
                    if sim_score >= drug_sim_threshold:
                        # Add similarity edge (bidirectional)
                        src_nodes.append(node_map[drug_i])
                        dst_nodes.append(node_map[drug_j])
                        edge_weights.append(float(sim_score))
                        # Use binding as default relation for similarity edges
                        edge_types.append(RelationType.BINDING)
                        edge_type_indices.append(relation_type_to_idx[RelationType.BINDING])
    
    # Add protein similarity edges if provided
    if protein_similarity is not None:
        print("Adding protein similarity edges...")
        protein_ids = protein_similarity.columns
        
        for i, protein_i in enumerate(protein_ids):
            if protein_i not in node_map:
                continue
                
            for j, protein_j in enumerate(protein_ids):
                if i != j and protein_j in node_map:
                    sim_score = protein_similarity.iloc[i, j]
                    
                    # Only add edges for similar proteins
                    if sim_score >= protein_sim_threshold:
                        # Add similarity edge (bidirectional)
                        src_nodes.append(node_map[protein_i])
                        dst_nodes.append(node_map[protein_j])
                        edge_weights.append(float(sim_score))
                        # Use binding as default relation for similarity edges
                        edge_types.append(RelationType.BINDING)
                        edge_type_indices.append(relation_type_to_idx[RelationType.BINDING])
    
    # Create edge index tensor
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    
    # Create edge weight tensor - ensure it's a 1D tensor with correct shape
    edge_weight = torch.tensor(edge_weights, dtype=torch.float).view(-1)
    
    # Create edge type tensor
    edge_type = torch.tensor(edge_type_indices, dtype=torch.long).view(-1)
    
    # Check for empty graph
    if edge_index.shape[1] == 0:
        print("Warning: Graph has no edges. Adding dummy edge to prevent errors.")
        # Add a dummy edge if no edges
        if len(all_nodes) >= 2:
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float)
            edge_type = torch.tensor([7, 7], dtype=torch.long)  # UNKNOWN relation
            edge_types = [RelationType.UNKNOWN, RelationType.UNKNOWN]
        else:
            raise ValueError("Cannot build graph: not enough nodes")
    
    # Add validation to ensure edge_weight matches the number of edges
    assert edge_index.shape[1] == edge_weight.size(0), \
        f"Edge count mismatch: edge_index has {edge_index.shape[1]} edges but edge_weight has {edge_weight.size(0)} values"
    
    # Extract node indices by type
    compound_indices = torch.tensor([node_map[c] for c in compounds if c in node_map], dtype=torch.long)
    target_indices = torch.tensor([node_map[t] for t in targets if t in node_map], dtype=torch.long)
    
    # Create node features
    node_features = []
    
    # Feature dimension (default if no features provided)
    feature_dim = 128
    if compound_features and len(compound_features) > 0:
        first_feature = next(iter(compound_features.values()))
        feature_dim = first_feature.shape[0]
    
    for i in range(len(all_nodes)):
        node_id = reverse_node_map[i]
        
        if node_id in compounds and node_id in compound_features:
            # Use provided compound features
            node_features.append(compound_features[node_id])
        elif node_id in targets and target_features and node_id in target_features:
            # Use provided target features
            node_features.append(target_features[node_id])
        else:
            # Create random features for nodes without provided features
            node_features.append(torch.randn(feature_dim))
    
    # Stack node features
    x = torch.stack(node_features)
    
    # Create node type tensor (0 for compounds, 1 for targets)
    node_type = torch.zeros(len(all_nodes), dtype=torch.long)
    for i, node_id in reverse_node_map.items():
        if node_id in targets:
            node_type[i] = 1
    
    # Create PyG Data object with semantic information
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_type=edge_type,
        node_type=node_type,
        compound_indices=compound_indices,
        target_indices=target_indices,
        num_relations=8  # Number of relation types defined
    )
    
    # Store relation types for each edge for later use
    relation_types = {i: edge_types[i] for i in range(len(edge_types))}
    
    print(f"Graph built with {data.x.shape[0]} nodes and {data.edge_index.shape[1]} edges")
    print(f"  - {len(compounds)} compounds")
    print(f"  - {len(targets)} targets")
    print(f"  - {len(set(edge_type_indices))} relation types")
    
    return data, node_map, reverse_node_map, relation_types
