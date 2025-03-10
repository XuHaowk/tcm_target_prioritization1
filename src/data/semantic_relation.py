"""
Semantic relation module for TCM target prioritization

This module defines semantic relationship types and their properties for TCM compounds and targets.
"""
import enum
import numpy as np

class RelationType(enum.Enum):
    """Enumeration of semantic relationship types between compounds and targets"""
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
        """Convert string to relation type"""
        relation_str = relation_str.lower() if relation_str else ""
        
        for relation in cls:
            if relation.value == relation_str:
                return relation
        
        # Map similar terms to standard types
        activation_terms = ["activate", "upregulate", "induce", "enhance", "increase"]
        inhibition_terms = ["inhibit", "downregulate", "suppress", "decrease", "reduce", "block"]
        binding_terms = ["bind", "interact", "complex"]
        modulation_terms = ["modulate", "regulate", "affect", "alter", "change"]
        substrate_terms = ["substrate", "metabolize", "cleave"]
        transport_terms = ["transport", "translocate", "shuttle", "carry"]
        indirect_terms = ["indirect", "secondary", "downstream"]
        
        for term in activation_terms:
            if term in relation_str:
                return cls.ACTIVATION
                
        for term in inhibition_terms:
            if term in relation_str:
                return cls.INHIBITION
                
        for term in binding_terms:
            if term in relation_str:
                return cls.BINDING
                
        for term in modulation_terms:
            if term in relation_str:
                return cls.MODULATION
                
        for term in substrate_terms:
            if term in relation_str:
                return cls.SUBSTRATE
                
        for term in transport_terms:
            if term in relation_str:
                return cls.TRANSPORT
                
        for term in indirect_terms:
            if term in relation_str:
                return cls.INDIRECT_REGULATION
                
        return cls.UNKNOWN


class SemanticRelationHandler:
    """Handler for semantic relationships"""
    
    def __init__(self):
        """Initialize relation handler with relation properties"""
        # Relation importance weights (how informative each relation type is)
        self.relation_weights = {
            RelationType.ACTIVATION: 0.9,      # Very informative
            RelationType.INHIBITION: 0.9,      # Very informative
            RelationType.MODULATION: 0.7,      # Moderately informative
            RelationType.BINDING: 0.5,         # Less informative
            RelationType.SUBSTRATE: 0.7,       # Moderately informative
            RelationType.TRANSPORT: 0.6,       # Moderately informative
            RelationType.INDIRECT_REGULATION: 0.4, # Less informative
            RelationType.UNKNOWN: 0.3          # Least informative
        }
        
        # Relation compatibility matrix (how compatible different relations are)
        # Higher values mean more compatible/similar relation types
        self.compatibility_matrix = {
            # Activation is compatible with activation, less with modulation, opposite of inhibition
            RelationType.ACTIVATION: {
                RelationType.ACTIVATION: 1.0,
                RelationType.INHIBITION: -0.8,
                RelationType.MODULATION: 0.5,
                RelationType.BINDING: 0.3,
                RelationType.SUBSTRATE: 0.2,
                RelationType.TRANSPORT: 0.1,
                RelationType.INDIRECT_REGULATION: 0.3,
                RelationType.UNKNOWN: 0.0
            },
            # Inhibition is compatible with inhibition, less with modulation, opposite of activation
            RelationType.INHIBITION: {
                RelationType.ACTIVATION: -0.8,
                RelationType.INHIBITION: 1.0,
                RelationType.MODULATION: 0.5,
                RelationType.BINDING: 0.3,
                RelationType.SUBSTRATE: 0.2,
                RelationType.TRANSPORT: 0.1,
                RelationType.INDIRECT_REGULATION: 0.3,
                RelationType.UNKNOWN: 0.0
            },
            # Other relations defined similarly
            RelationType.MODULATION: {
                RelationType.ACTIVATION: 0.5,
                RelationType.INHIBITION: 0.5,
                RelationType.MODULATION: 1.0,
                RelationType.BINDING: 0.4,
                RelationType.SUBSTRATE: 0.3,
                RelationType.TRANSPORT: 0.2,
                RelationType.INDIRECT_REGULATION: 0.5,
                RelationType.UNKNOWN: 0.1
            },
            RelationType.BINDING: {
                RelationType.ACTIVATION: 0.3,
                RelationType.INHIBITION: 0.3,
                RelationType.MODULATION: 0.4,
                RelationType.BINDING: 1.0,
                RelationType.SUBSTRATE: 0.4,
                RelationType.TRANSPORT: 0.3,
                RelationType.INDIRECT_REGULATION: 0.2,
                RelationType.UNKNOWN: 0.1
            },
            RelationType.SUBSTRATE: {
                RelationType.ACTIVATION: 0.2,
                RelationType.INHIBITION: 0.2,
                RelationType.MODULATION: 0.3,
                RelationType.BINDING: 0.4,
                RelationType.SUBSTRATE: 1.0,
                RelationType.TRANSPORT: 0.5,
                RelationType.INDIRECT_REGULATION: 0.3,
                RelationType.UNKNOWN: 0.1
            },
            RelationType.TRANSPORT: {
                RelationType.ACTIVATION: 0.1,
                RelationType.INHIBITION: 0.1,
                RelationType.MODULATION: 0.2,
                RelationType.BINDING: 0.3,
                RelationType.SUBSTRATE: 0.5,
                RelationType.TRANSPORT: 1.0,
                RelationType.INDIRECT_REGULATION: 0.2,
                RelationType.UNKNOWN: 0.1
            },
            RelationType.INDIRECT_REGULATION: {
                RelationType.ACTIVATION: 0.3,
                RelationType.INHIBITION: 0.3,
                RelationType.MODULATION: 0.5,
                RelationType.BINDING: 0.2,
                RelationType.SUBSTRATE: 0.3,
                RelationType.TRANSPORT: 0.2,
                RelationType.INDIRECT_REGULATION: 1.0,
                RelationType.UNKNOWN: 0.1
            },
            RelationType.UNKNOWN: {
                RelationType.ACTIVATION: 0.0,
                RelationType.INHIBITION: 0.0,
                RelationType.MODULATION: 0.1,
                RelationType.BINDING: 0.1,
                RelationType.SUBSTRATE: 0.1,
                RelationType.TRANSPORT: 0.1,
                RelationType.INDIRECT_REGULATION: 0.1,
                RelationType.UNKNOWN: 1.0
            }
        }
    
    def get_relation_weight(self, relation_type):
        """Get weight for a relation type"""
        if isinstance(relation_type, str):
            relation_type = RelationType.from_string(relation_type)
        
        return self.relation_weights.get(relation_type, self.relation_weights[RelationType.UNKNOWN])
    
    def get_relation_compatibility(self, relation1, relation2):
        """Get compatibility between two relation types"""
        if isinstance(relation1, str):
            relation1 = RelationType.from_string(relation1)
        if isinstance(relation2, str):
            relation2 = RelationType.from_string(relation2)
        
        return self.compatibility_matrix.get(relation1, {}).get(
            relation2, 0.0)
    
    def calculate_path_similarity(self, path_relations, decay_factor=0.8):
        """
        Calculate semantic similarity along a path based on relation compatibility
        
        Args:
            path_relations: List of relations along the path
            decay_factor: Factor for decaying influence with path length
            
        Returns:
            Path similarity score
        """
        if not path_relations or len(path_relations) < 2:
            return 0.0
            
        # Calculate product of relation compatibilities along the path
        similarity = 1.0
        for i in range(len(path_relations) - 1):
            rel1 = path_relations[i]
            rel2 = path_relations[i + 1]
            compatibility = self.get_relation_compatibility(rel1, rel2)
            
            # For negative compatibilities (opposite relations), use absolute value but penalize
            if compatibility < 0:
                similarity *= (abs(compatibility) * 0.5)
            else:
                similarity *= compatibility
        
        # Apply decay factor based on path length
        path_length = len(path_relations)
        similarity *= (decay_factor ** (path_length - 1))
        
        return similarity
    
    def calculate_semantic_similarity(self, relations1, relations2):
        """
        Calculate semantic similarity between two sets of relations
        
        Args:
            relations1: First set of relations
            relations2: Second set of relations
            
        Returns:
            Semantic similarity score
        """
        if not relations1 or not relations2:
            return 0.0
            
        # Calculate maximum compatibility between any pair of relations
        max_compatibility = 0.0
        for rel1 in relations1:
            for rel2 in relations2:
                compatibility = self.get_relation_compatibility(rel1, rel2)
                max_compatibility = max(max_compatibility, abs(compatibility))
        
        return max_compatibility
    
    def get_relation_embedding(self, relation_type, embedding_dim=8):
        """
        Get embedding vector for a relation type
        
        Args:
            relation_type: Relation type
            embedding_dim: Dimension of embedding vector
            
        Returns:
            Embedding vector
        """
        if isinstance(relation_type, str):
            relation_type = RelationType.from_string(relation_type)
            
        # Create fixed embeddings for each relation type
        # These could also be learned during training
        if relation_type == RelationType.ACTIVATION:
            base = np.array([1.0, 0.2, 0.0, 0.0, 0.5, 0.3, 0.0, 0.1])
        elif relation_type == RelationType.INHIBITION:
            base = np.array([-1.0, 0.2, 0.0, 0.0, 0.5, 0.3, 0.0, 0.1])
        elif relation_type == RelationType.MODULATION:
            base = np.array([0.5, 1.0, 0.2, 0.0, 0.2, 0.3, 0.0, 0.1])
        elif relation_type == RelationType.BINDING:
            base = np.array([0.0, 0.2, 1.0, 0.0, 0.0, 0.3, 0.0, 0.1])
        elif relation_type == RelationType.SUBSTRATE:
            base = np.array([0.0, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0, 0.1])
        elif relation_type == RelationType.TRANSPORT:
            base = np.array([0.0, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0, 0.1])
        elif relation_type == RelationType.INDIRECT_REGULATION:
            base = np.array([0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1])
        else:  # UNKNOWN
            base = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1])
            
        # Ensure correct dimensionality
        if embedding_dim == 8:
            return base
        elif embedding_dim < 8:
            return base[:embedding_dim]
        else:
            # Pad with zeros
            padded = np.zeros(embedding_dim)
            padded[:8] = base
            return padded
