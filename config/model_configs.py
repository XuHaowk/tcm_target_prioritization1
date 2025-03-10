"""
模型配置模块

这个模块提供了不同模型架构的配置。
"""

# 基础RGCN的配置
BASE_RGCN_CONFIG = {
    'in_dim': None,  # 将在运行时设置
    'hidden_dim': 256,
    'out_dim': 128,
    'num_relations': None,  # 将在运行时设置
    'num_bases': None,  # 将在运行时设置
    'dropout': 0.3
}

# 注意力增强RGCN的配置
ATTENTION_RGCN_CONFIG = {
    'in_dim': None,  # 将在运行时设置
    'hidden_dims': [256, 256, 256],
    'out_dim': 128,
    'num_relations': None,  # 将在运行时设置
    'num_bases': None,  # 将在运行时设置
    'dropout': 0.3,
    'residual': True,
    'layer_norm': True
}

# 边特征RGCN的配置
EDGE_FEATURE_RGCN_CONFIG = {
    'in_dim': None,  # 将在运行时设置
    'hidden_dim': 256,
    'out_dim': 128,
    'num_relations': None,  # 将在运行时设置
    'edge_dim': None,  # 将在运行时设置
    'num_bases': None,  # 将在运行时设置
    'dropout': 0.3
}

# 多尺度RGCN的配置
MULTISCALE_RGCN_CONFIG = {
    'in_dim': None,  # 将在运行时设置
    'hidden_dim': 256,
    'out_dim': 128,
    'num_relations': None,  # 将在运行时设置
    'num_layers': 4,
    'num_bases': None,  # 将在运行时设置
    'dropout': 0.3
}

# 图池化RGCN的配置
GRAPH_POOL_RGCN_CONFIG = {
    'in_dim': None,  # 将在运行时设置
    'hidden_dim': 256,
    'out_dim': 128,
    'num_relations': None,  # 将在运行时设置
    'num_bases': None,  # 将在运行时设置
    'dropout': 0.3,
    'pooling': 'attention'  # 'sum', 'mean', 'max', 'attention'
}

# 训练配置
TRAINING_CONFIG = {
    'num_epochs': 150,
    'batch_size': 64,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'margin': 0.5,
    'neg_samples': 2,
    'patience': 15
}

# 优先级排序配置
PRIORITY_CONFIG = {
    'embedding_weight': 0.05,
    'importance_weight': 0.75,
    'drug_sim_weight': 0.075,
    'protein_sim_weight': 0.075,
    'semantic_weight': 0.05,
    'structure_weight': 0.1,
    'top_k': 10
}

# 根据模型类型获取配置
def get_model_config(model_type):
    """
    根据模型类型获取配置
    
    参数:
        model_type: 模型类型
        
    返回:
        模型配置字典
    """
    if model_type == 'attention_rgcn':
        return ATTENTION_RGCN_CONFIG
    elif model_type == 'edge_feature_rgcn':
        return EDGE_FEATURE_RGCN_CONFIG
    elif model_type == 'multiscale_rgcn':
        return MULTISCALE_RGCN_CONFIG
    elif model_type == 'graph_pool_rgcn':
        return GRAPH_POOL_RGCN_CONFIG
    else:  # 默认为基础RGCN
        return BASE_RGCN_CONFIG
