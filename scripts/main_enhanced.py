#!/usr/bin/env python3
"""
矽肺靶点优先级排序系统 - 增强版主脚本

这个脚本提供了增强版矽肺靶点优先级排序系统的主要功能，包括：
1. 使用多种图神经网络架构
2. 增强的训练过程
3. 改进的评估指标
4. 图结构感知的得分计算
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 将项目根目录添加到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
from src.data.graph_builder import build_semantic_graph
from src.models import (
    RGCN, AttentionRGCN, EdgeFeatureRGCN, MultiscaleRGCN, GraphPoolRGCN
)
from src.training import EnhancedTrainer
from src.evaluation import (
    calculate_enhanced_metrics, 
    plot_enhanced_validation_metrics,
    create_enhanced_visualization
)

def add_model_type_args(parser):
    """添加模型类型相关参数"""
    parser.add_argument('--model_type', type=str, default='rgcn',
                       choices=['rgcn', 'attention_rgcn', 'edge_feature_rgcn', 
                                'multiscale_rgcn', 'graph_pool_rgcn'],
                       help='Model architecture to use')
    parser.add_argument('--structure_weight', type=float, default=0.1,
                       help='Weight for graph structure score')
    parser.add_argument('--use_layer_norm', action='store_true',
                       help='Use layer normalization')
    parser.add_argument('--use_residual', action='store_true',
                       help='Use residual connections')
    parser.add_argument('--hidden_dims', type=str, default='256,256,256',
                       help='Comma-separated list of hidden dimensions')
    return parser

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Enhanced TCM Target Prioritization')
    
    # 数据参数
    parser.add_argument('--kg_data', type=str, default='data/raw/kg_data_extended.csv',
                       help='Knowledge graph data file')
    parser.add_argument('--db_data', type=str, default='data/raw/database_data_extended.csv',
                       help='Database data file')
    parser.add_argument('--disease_importance', type=str, default='data/raw/disease_importance_extended.csv',
                       help='Disease importance data file')
    parser.add_argument('--validated_data', type=str, default='data/raw/validated_interactions.csv',
                       help='Validated interactions data file')
    parser.add_argument('--drug_similarity', type=str, default=None,
                       help='Drug similarity matrix file')
    parser.add_argument('--protein_similarity', type=str, default=None,
                       help='Protein similarity matrix file')
    parser.add_argument('--semantic_similarities', type=str, default=None,
                       help='Semantic similarities file')
    parser.add_argument('--generate_samples', action='store_true',
                       help='Generate sample data instead of loading from files')
    
    # 模型参数
    parser.add_argument('--load_model', action='store_true',
                       help='Load pre-trained model')
    parser.add_argument('--model_path', type=str, default='results/models/rgcn_model.pt',
                       help='Path to pre-trained model')
    parser.add_argument('--save_model', action='store_true',
                       help='Save trained model')
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU instead of GPU')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--margin', type=float, default=0.5,
                       help='Margin for contrastive loss')
    parser.add_argument('--neg_samples', type=int, default=2,
                       help='Number of negative samples per positive sample')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience for early stopping')
    
    # 优先级排序参数
    parser.add_argument('--use_drug_similarity', action='store_true',
                       help='Use drug similarity for prioritization')
    parser.add_argument('--use_protein_similarity', action='store_true',
                       help='Use protein similarity for prioritization')
    parser.add_argument('--use_semantic_similarity', action='store_true',
                       help='Use semantic similarity for prioritization')
    parser.add_argument('--embedding_weight', type=float, default=0.05,
                       help='Weight for embedding similarity')
    parser.add_argument('--importance_weight', type=float, default=0.75,
                       help='Weight for disease importance')
    parser.add_argument('--drug_sim_weight', type=float, default=0.075,
                       help='Weight for drug similarity')
    parser.add_argument('--protein_sim_weight', type=float, default=0.075,
                       help='Weight for protein similarity')
    parser.add_argument('--semantic_weight', type=float, default=0.05,
                       help='Weight for semantic similarity')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of top targets to return')
    
    # 验证参数
    parser.add_argument('--validate', action='store_true',
                       help='Validate model')
    
    # 添加模型类型参数
    parser = add_model_type_args(parser)
    
    return parser.parse_args()

def load_data(args):
    """
    加载数据
    
    参数:
        args: 命令行参数
        
    返回:
        元组 (kg_data, db_data, disease_importance, validated_data, 
              drug_similarity, protein_similarity, semantic_similarities)
    """
    # 检查是否设置了generate_samples（默认为False）
    generate_samples = getattr(args, 'generate_samples', False)
    
    if generate_samples:
        # 生成样本数据（未实现）
        print("Generating sample data not implemented in this version.")
        return None, None, None, None, None, None, None
    
    # 加载知识图谱数据
    kg_data = pd.read_csv(args.kg_data)
    print(f"Loaded knowledge graph with {len(kg_data)} relations")
    
    # 加载数据库数据
    db_data = pd.read_csv(args.db_data)
    print(f"Loaded database with {len(db_data)} compounds")
    
    # 加载疾病重要性数据
    disease_importance = pd.read_csv(args.disease_importance)
    print(f"Loaded disease importance data with {len(disease_importance)} entries")
    
    # 加载验证交互（如果提供）
    validated_data = None
    if args.validated_data and os.path.exists(args.validated_data):
        validated_data = pd.read_csv(args.validated_data)
        print(f"Loaded {len(validated_data)} validated interactions")
    
    # 加载药物相似性矩阵（如果提供）
    drug_similarity = None
    if args.drug_similarity and os.path.exists(args.drug_similarity):
        print(f"Loading drug similarity matrix from {args.drug_similarity}")
        drug_similarity = pd.read_csv(args.drug_similarity, index_col=0)
    
    # 加载蛋白质相似性矩阵（如果提供）
    protein_similarity = None
    if args.protein_similarity and os.path.exists(args.protein_similarity):
        print(f"Loading protein similarity matrix from {args.protein_similarity}")
        protein_similarity = pd.read_csv(args.protein_similarity, index_col=0)
    
    # 加载语义相似性（如果提供）
    semantic_similarities = None
    if args.semantic_similarities and os.path.exists(args.semantic_similarities):
        print(f"Loading semantic similarities from {args.semantic_similarities}")
        semantic_similarities = pd.read_csv(args.semantic_similarities)
    
    return kg_data, db_data, disease_importance, validated_data, drug_similarity, protein_similarity, semantic_similarities

def extract_important_targets(disease_importance, node_map):
    """
    提取重要靶点
    
    参数:
        disease_importance: 疾病重要性DataFrame
        node_map: 节点ID到索引的映射
        
    返回:
        重要靶点索引列表
    """
    # 检查列名以查找靶点标识符列
    target_id_columns = ['target_id', 'target', 'protein_id', 'protein', 'gene_id', 'gene']
    target_col = None
    
    for col in target_id_columns:
        if col in disease_importance.columns:
            target_col = col
            break
    
    if target_col is None:
        print("WARNING: Could not find target identifier column in disease_importance dataframe.")
        print(f"Available columns: {list(disease_importance.columns)}")
        print("Using first column as target identifier.")
        target_col = disease_importance.columns[0]
    
    print(f"Using '{target_col}' as target identifier column.")
    
    # 提取靶点索引
    important_targets = []
    
    for _, row in disease_importance.iterrows():
        target_id = row[target_col]
        
        if target_id in node_map:
            target_idx = node_map[target_id]
            important_targets.append(target_idx)
    
    return important_targets

def calculate_graph_structure_score(data, compound_idx, target_idx, node_map, reverse_node_map,
                                   embeddings, alpha=0.5, max_hops=2):
    """
    计算基于图结构的得分
    
    参数:
        data: PyG数据对象
        compound_idx: 化合物索引
        target_idx: 靶点索引
        node_map: 节点ID到索引的映射
        reverse_node_map: 索引到节点ID的映射
        embeddings: 节点嵌入
        alpha: 随步长衰减因子
        max_hops: 最大跳数
        
    返回:
        结构得分
    """
    # 提取边索引和类型
    edge_index = data.edge_index.cpu().numpy()
    edge_type = data.edge_type.cpu().numpy() if hasattr(data, 'edge_type') else None
    
    # 构建邻接表
    adj_list = {}
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        rel = edge_type[i] if edge_type is not None else None
        
        if src not in adj_list:
            adj_list[src] = []
        adj_list[src].append((dst, rel))
    
    # 使用BFS查找路径
    def find_paths(start, end, max_depth):
        visited = set()
        queue = [(start, [], 0)]  # (node, path, depth)
        paths = []
        
        while queue:
            node, path, depth = queue.pop(0)
            
            if depth > max_depth:
                continue
            
            if node == end:
                paths.append(path + [node])
                continue
            
            if node in visited:
                continue
            
            visited.add(node)
            
            if node in adj_list:
                for neighbor, rel in adj_list[node]:
                    queue.append((neighbor, path + [node], depth + 1))
        
        return paths
    
    # 查找从化合物到靶点的路径
    paths = find_paths(compound_idx, target_idx, max_hops)
    
    # 计算路径得分
    path_scores = []
    
    for path in paths:
        # 路径长度
        path_length = len(path) - 1
        
        # 路径得分 (衰减因子 * 路径嵌入相似度)
        decay = alpha ** path_length
        
        # 计算路径上节点嵌入的平均相似度
        sim_sum = 0.0
        for i in range(path_length):
            node1 = path[i]
            node2 = path[i + 1]
            
            emb1 = embeddings[node1].cpu().numpy()
            emb2 = embeddings[node2].cpu().numpy()
            
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            sim_sum += sim
        
        avg_sim = sim_sum / path_length if path_length > 0 else 0.0
        
        # 计算最终路径得分
        path_score = decay * avg_sim
        path_scores.append(path_score)
    
    # 如果找到路径，返回最高得分
    if path_scores:
        return max(path_scores)
    
    # 如果没有路径，返回0分
    return 0.0

def calculate_target_priorities_with_graph_structure(embeddings, data, compound_id, node_map, reverse_node_map, 
                                 disease_importance, drug_similarity=None, protein_similarity=None,
                                 semantic_similarities=None,
                                 embedding_weight=0.05, importance_weight=0.75,
                                 drug_sim_weight=0.075, protein_sim_weight=0.075,
                                 semantic_weight=0.05, structure_weight=0.1, k=10, save_results=True):
    """
    使用图结构信息计算靶点优先级
    
    参数:
        embeddings: 节点嵌入
        data: PyG数据对象
        compound_id: 化合物ID
        node_map: 节点ID到索引的映射
        reverse_node_map: 索引到节点ID的映射
        disease_importance: 疾病重要性DataFrame
        drug_similarity: 药物相似性DataFrame
        protein_similarity: 蛋白质相似性DataFrame
        semantic_similarities: 语义相似性DataFrame
        embedding_weight: 嵌入相似度权重
        importance_weight: 疾病重要性权重
        drug_sim_weight: 药物相似度权重
        protein_sim_weight: 蛋白质相似度权重
        semantic_weight: 语义相似度权重
        structure_weight: 图结构得分权重
        k: 返回的前k个靶点数量
        save_results: 是否保存结果
        
    返回:
        靶点优先级DataFrame
    """
    # 归一化权重
    total_weight = (embedding_weight + importance_weight + drug_sim_weight + 
                   protein_sim_weight + semantic_weight + structure_weight)
    
    embedding_weight /= total_weight
    importance_weight /= total_weight
    drug_sim_weight /= total_weight
    protein_sim_weight /= total_weight
    semantic_weight /= total_weight
    structure_weight /= total_weight
    
    print(f"Normalized weights: Embedding={embedding_weight:.4f}, Importance={importance_weight:.4f}, " +
          f"Drug Sim={drug_sim_weight:.4f}, Protein Sim={protein_sim_weight:.4f}, " +
          f"Semantic={semantic_weight:.4f}, Structure={structure_weight:.4f}")
    
    # 提取靶点索引
    target_indices = data.target_indices.cpu().numpy()
    
    # 获取化合物索引
    if compound_id not in node_map:
        raise ValueError(f"Compound {compound_id} not found in graph")
    compound_idx = node_map[compound_id]
    
    # 提取化合物和靶点嵌入
    compound_embedding = embeddings[compound_idx].cpu().numpy()
    
    # 计算所有靶点的嵌入相似度
    similarities = {}
    
    for target_idx in target_indices:
        target_id = reverse_node_map[target_idx]
        target_embedding = embeddings[target_idx].cpu().numpy()
        
        # 计算余弦相似度
        sim = np.dot(compound_embedding, target_embedding) / \
              (np.linalg.norm(compound_embedding) * np.linalg.norm(target_embedding))
        
        similarities[target_id] = sim
    
    # 寻找疾病重要性数据中的靶点和重要性列
    target_col = None
    importance_col = None
    
    # 检查靶点列
    target_columns = ['target_id', 'target', 'protein_id', 'protein', 'gene_id', 'gene']
    for col in target_columns:
        if col in disease_importance.columns:
            target_col = col
            break
    
    # 检查重要性列
    importance_columns = ['importance_score', 'importance', 'score', 'weight', 'priority']
    for col in importance_columns:
        if col in disease_importance.columns:
            importance_col = col
            break
    
    # 计算所有靶点的疾病重要性得分
    importance_scores = {}
    
    if target_col and importance_col:
        for target_idx in target_indices:
            target_id = reverse_node_map[target_idx]
            
            # 从数据中获取重要性得分
            mask = disease_importance[target_col] == target_id
            if mask.any():
                importance = disease_importance.loc[mask, importance_col].values[0]
                importance_scores[target_id] = float(importance)
            else:
                importance_scores[target_id] = 0.0
    else:
        # 如果找不到合适的列，使用默认重要性
        for target_idx in target_indices:
            target_id = reverse_node_map[target_idx]
            importance_scores[target_id] = 1.0
    
    # 应用非线性变换增强重要性得分的对比度
    max_importance = max(importance_scores.values()) if importance_scores else 1.0
    for target_id in importance_scores:
        # 归一化重要性
        norm_importance = importance_scores[target_id] / max_importance
        # 应用幂变换
        adjusted_importance = np.power(norm_importance, 1.5)  # 幂变换
        importance_scores[target_id] = adjusted_importance
    
    # 计算基于药物的相似度得分
    drug_sim_scores = {}
    
    if drug_similarity is not None and drug_sim_weight > 0:
        # 检查化合物是否在药物相似性矩阵中
        if compound_id not in drug_similarity.index:
            print(f"WARNING: Compound '{compound_id}' not found in drug similarity matrix. Using default scores of 0.0.")
            # 为所有靶点返回零分
            for target_idx in target_indices:
                target_id = reverse_node_map[target_idx]
                drug_sim_scores[target_id] = 0.0
        else:
            # 获取所有靶点连接到的与查询化合物相似的化合物
            for target_idx in target_indices:
                target_id = reverse_node_map[target_idx]
                
                # 获取与查询化合物相似的所有化合物
                similar_compounds = []
                for other_compound in drug_similarity.columns:
                    # 跳过索引中不存在或化合物本身的情况
                    if other_compound != compound_id and other_compound in drug_similarity.index:
                        try:
                            sim = drug_similarity.loc[compound_id, other_compound]
                            if sim > 0:  # 只考虑正相似度
                                similar_compounds.append((other_compound, sim))
                        except KeyError:
                            # 跳过访问相似度时出现的任何问题
                            continue
                
                # 检查靶点是否连接到任何相似化合物
                score = 0.0
                
                for other_compound, sim in similar_compounds:
                    if other_compound in node_map:
                        other_idx = node_map[other_compound]
                        
                        # 检查相似化合物和靶点之间是否有边
                        edge_indices = data.edge_index.cpu().numpy()
                        for i in range(edge_indices.shape[1]):
                            if (edge_indices[0, i] == other_idx and edge_indices[1, i] == target_idx) or \
                               (edge_indices[0, i] == target_idx and edge_indices[1, i] == other_idx):
                                # 通过化合物之间的相似度加权
                                score += sim
                                break
                
                drug_sim_scores[target_id] = score
    
    # 规范化药物相似度得分
    if drug_sim_scores:
        max_drug_sim = max(drug_sim_scores.values())
        if max_drug_sim > 0:
            for target_id in drug_sim_scores:
                drug_sim_scores[target_id] /= max_drug_sim
    
    # 计算基于蛋白质的相似度得分
    protein_sim_scores = {}
    
    if protein_similarity is not None and protein_sim_weight > 0:
        # 首先，找出已知与化合物交互的靶点
        known_targets = []
        edge_indices = data.edge_index.cpu().numpy()
        
        for i in range(edge_indices.shape[1]):
            if edge_indices[0, i] == compound_idx:
                target_idx = edge_indices[1, i]
                if target_idx in target_indices:
                    known_targets.append(reverse_node_map[target_idx])
            elif edge_indices[1, i] == compound_idx:
                target_idx = edge_indices[0, i]
                if target_idx in target_indices:
                    known_targets.append(reverse_node_map[target_idx])
        
        # 计算基于蛋白质相似度的得分
        for target_idx in target_indices:
            target_id = reverse_node_map[target_idx]
            
            # 如果靶点已知与化合物交互，则跳过
            if target_id in known_targets:
                protein_sim_scores[target_id] = 1.0
                continue
            
            # 计算与已知靶点的蛋白质相似度
            max_sim = 0.0
            
            for known_target in known_targets:
                # 处理靶点ID可能不在相似度矩阵中的情况
                if known_target in protein_similarity.index and target_id in protein_similarity.columns:
                    try:
                        sim = protein_similarity.loc[known_target, target_id]
                        max_sim = max(max_sim, sim)
                    except KeyError:
                        continue
            
            protein_sim_scores[target_id] = max_sim
    
    # 计算语义相似度得分
    semantic_sim_scores = {}
    
    if semantic_similarities is not None and semantic_weight > 0:
        # 找出语义相似度中的化合物列
        compound_col = None
        compound_columns = ['compound_id', 'compound', 'drug_id', 'drug']
        
        for col in compound_columns:
            if col in semantic_similarities.columns:
                compound_col = col
                break
        
        if compound_col is None and len(semantic_similarities.columns) > 0:
            compound_col = semantic_similarities.columns[0]
            print(f"WARNING: Could not find compound column in semantic similarities. Using '{compound_col}'.")
        
        # 找出语义相似度中的靶点列
        target_col = None
        target_columns = ['target_id', 'target', 'protein_id', 'protein']
        
        for col in target_columns:
            if col in semantic_similarities.columns:
                target_col = col
                break
        
        if target_col is None and len(semantic_similarities.columns) > 1:
            target_col = semantic_similarities.columns[1]
            print(f"WARNING: Could not find target column in semantic similarities. Using '{target_col}'.")
        
        # 找出语义相似度中的相似度列
        sim_col = None
        sim_columns = ['semantic_similarity', 'similarity', 'sim_score', 'score']
        
        for col in sim_columns:
            if col in semantic_similarities.columns:
                sim_col = col
                break
        
        if sim_col is None and len(semantic_similarities.columns) > 2:
            sim_col = semantic_similarities.columns[2]
            print(f"WARNING: Could not find similarity column in semantic similarities. Using '{sim_col}'.")
        
        # 计算语义相似度得分
        if compound_col is not None and target_col is not None and sim_col is not None:
            for target_idx in target_indices:
                target_id = reverse_node_map[target_idx]
                
                # 检查此化合物-靶点对是否存在语义相似度
                mask = (semantic_similarities[compound_col] == compound_id) & \
                      (semantic_similarities[target_col] == target_id)
                
                if mask.any():
                    semantic_sim_scores[target_id] = semantic_similarities.loc[mask, sim_col].values[0]
                else:
                    semantic_sim_scores[target_id] = 0.0
    
    # 计算图结构得分
    structure_scores = {}
    
    if structure_weight > 0:
        for target_idx in target_indices:
            target_id = reverse_node_map[target_idx]
            
            # 计算图结构得分
            structure_score = calculate_graph_structure_score(
                data, compound_idx, target_idx, node_map, reverse_node_map, embeddings
            )
            
            structure_scores[target_id] = structure_score
    
    # 组合所有得分
    final_scores = {}
    
    for target_idx in target_indices:
        target_id = reverse_node_map[target_idx]
        
        # 获取各组成部分的得分（如果不可用则默认为0）
        emb_sim = similarities.get(target_id, 0.0)
        imp_score = importance_scores.get(target_id, 0.0)
        drug_score = drug_sim_scores.get(target_id, 0.0) if drug_sim_weight > 0 else 0.0
        protein_score = protein_sim_scores.get(target_id, 0.0) if protein_sim_weight > 0 else 0.0
        semantic_score = semantic_sim_scores.get(target_id, 0.0) if semantic_weight > 0 else 0.0
        structure_score = structure_scores.get(target_id, 0.0) if structure_weight > 0 else 0.0
        
        # 加权组合
        final_score = (embedding_weight * emb_sim +
                      importance_weight * imp_score +
                      drug_sim_weight * drug_score +
                      protein_sim_weight * protein_score +
                      semantic_weight * semantic_score +
                      structure_weight * structure_score)
        
        final_scores[target_id] = final_score
    
    # 按最终得分排序靶点
    sorted_targets = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 创建结果DataFrame
    results = []
    
    for target_id, final_score in sorted_targets[:k]:
        # 获取各组成部分得分
        emb_sim = similarities.get(target_id, 0.0)
        imp_score = importance_scores.get(target_id, 0.0)
        drug_score = drug_sim_scores.get(target_id, 0.0) if drug_sim_weight > 0 else 0.0
        protein_score = protein_sim_scores.get(target_id, 0.0) if protein_sim_weight > 0 else 0.0
        semantic_score = semantic_sim_scores.get(target_id, 0.0) if semantic_weight > 0 else 0.0
        structure_score = structure_scores.get(target_id, 0.0) if structure_weight > 0 else 0.0
        
        results.append({
            'compound_id': compound_id,
            'target_id': target_id,
            'final_score': final_score,
            'embedding_similarity': emb_sim,
            'importance_score': imp_score,
            'drug_similarity_score': drug_score,
            'protein_similarity_score': protein_score,
            'semantic_similarity_score': semantic_score,
            'structure_score': structure_score
        })
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 保存结果
    if save_results:
        os.makedirs('results/prioritization', exist_ok=True)
        df.to_csv(f'results/prioritization/target_priorities_{compound_id}.csv', index=False)
        print(f"Results saved to results/prioritization/target_priorities_{compound_id}.csv")
    
    return df

def validate_enhanced_predictions(predictions, validated_data):
    """
    验证预测结果
    
    参数:
        predictions: 预测结果DataFrame
        validated_data: 验证数据DataFrame
    """
    try:
        # 确保输出目录存在
        os.makedirs('results/evaluation', exist_ok=True)
        
        # 标准化验证数据中的列名
        if 'compound_id' not in validated_data.columns and 'compound' in validated_data.columns:
            validated_data = validated_data.rename(columns={'compound': 'compound_id'})
        
        if 'target_id' not in validated_data.columns and 'target' in validated_data.columns:
            validated_data = validated_data.rename(columns={'target': 'target_id'})
        
        # 检查必要的列是否存在
        required_columns = ['compound_id', 'target_id']
        for col in required_columns:
            if col not in validated_data.columns:
                print(f"Warning: '{col}' column not found in validated data. Available columns: {validated_data.columns.tolist()}")
                print("Skipping validation due to missing columns.")
                return
            
            if col not in predictions.columns:
                print(f"Warning: '{col}' column not found in predictions. Available columns: {predictions.columns.tolist()}")
                print("Skipping validation due to missing columns.")
                return
        
        # 计算评估指标
        metrics = calculate_enhanced_metrics(predictions, validated_data)
        
        # 打印指标
        print("\n=== Validation Metrics ===")
        print(f"Precision-Recall AUC: {metrics['pr_auc']:.4f}" if not np.isnan(metrics['pr_auc']) else "Precision-Recall AUC: N/A")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}" if not np.isnan(metrics['roc_auc']) else "ROC AUC: N/A")
        print(f"Mean Reciprocal Rank: {metrics['mrr']:.4f}")
        
        for k in sorted([int(k.split('@')[1]) for k in metrics.keys() if k.startswith('hit@')]):
            print(f"Hit@{k}: {metrics[f'hit@{k}']:.4f}")
        
        # 绘制指标
        try:
            fig = plot_enhanced_validation_metrics(metrics, save_path='results/evaluation/validation_results.png')
            plt.close(fig)
            print("Validation metrics plot saved to results/evaluation/validation_results.png")
        except Exception as e:
            print(f"Warning: Failed to create validation metrics plot: {str(e)}")
        
        # 保存指标到CSV
        metrics_to_save = {
            'Metric': ['PR AUC', 'ROC AUC', 'MRR'] + [f'Hit@{k}' for k in sorted([int(k.split('@')[1]) for k in metrics.keys() if k.startswith('hit@')])],
            'Value': [
                metrics['pr_auc'] if not np.isnan(metrics['pr_auc']) else "N/A",
                metrics['roc_auc'] if not np.isnan(metrics['roc_auc']) else "N/A",
                metrics['mrr']
            ] + [metrics[f'hit@{k}'] for k in sorted([int(k.split('@')[1]) for k in metrics.keys() if k.startswith('hit@')])]
        }
        pd.DataFrame(metrics_to_save).to_csv('results/evaluation/metrics.csv', index=False)
        print("Validation metrics saved to results/evaluation/metrics.csv")
    
    except Exception as e:
        print(f"Warning: Validation failed with error: {str(e)}")
        print("Continuing without validation...")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载数据
    print("Loading data...")
    kg_data, db_data, disease_importance, validated_data, drug_similarity, protein_similarity, semantic_similarities = load_data(args)
    
    # 构建语义图
    print("Building semantic graph...")
    data, node_map, reverse_node_map, relation_types = build_semantic_graph(
        kg_data=kg_data,
        drug_similarity=drug_similarity if args.use_drug_similarity else None,
        protein_similarity=protein_similarity if args.use_protein_similarity else None
    )
    
    # 提取重要靶点
    important_targets = extract_important_targets(disease_importance, node_map)
    print(f"Found {len(important_targets)} important targets for training")
    
    # 设置设备
    use_cpu = getattr(args, 'cpu', False)
    device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')
    print(f"Using device: {device}")
    
    # 初始化模型
    in_dim = data.x.shape[1]
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')] if hasattr(args, 'hidden_dims') else [256, 256, 256]
    out_dim = 128
    
    # 使用不同类型的模型
    model_type = args.model_type if hasattr(args, 'model_type') else 'rgcn'
    
    if model_type == 'attention_rgcn':
        model = AttentionRGCN(
            in_dim, hidden_dims, out_dim, 
            num_relations=data.num_relations, 
            dropout=0.3,
            residual=args.use_residual if hasattr(args, 'use_residual') else True,
            layer_norm=args.use_layer_norm if hasattr(args, 'use_layer_norm') else True
        ).to(device)
    elif model_type == 'edge_feature_rgcn':
        model = EdgeFeatureRGCN(
            in_dim, hidden_dims[0], out_dim, 
            num_relations=data.num_relations, 
            dropout=0.3
        ).to(device)
    elif model_type == 'multiscale_rgcn':
        model = MultiscaleRGCN(
            in_dim, hidden_dims[0], out_dim, 
            num_relations=data.num_relations, 
            num_layers=len(hidden_dims),
            dropout=0.3
        ).to(device)
    elif model_type == 'graph_pool_rgcn':
        model = GraphPoolRGCN(
            in_dim, hidden_dims[0], out_dim, 
            num_relations=data.num_relations, 
            dropout=0.3,
            pooling='attention'
        ).to(device)
    else:  # 默认RGCN
        model = RGCN(
            in_dim, hidden_dims[0], out_dim, 
            num_relations=data.num_relations, 
            dropout=0.3
        ).to(device)
    
    print(f"Using model: {model.__class__.__name__}")
    
    # 训练或加载模型
    if args.load_model and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        
        # 生成嵌入
        with torch.no_grad():
            data = data.to(device)
            if hasattr(data, 'batch'):
                embeddings = model(data.x, data.edge_index, data.edge_type, data.batch, data.edge_weight)
            else:
                embeddings = model(data.x, data.edge_index, data.edge_type, data.edge_weight)
    else:
        print("Training model...")
        # 初始化增强训练器
        trainer = EnhancedTrainer(
            model=model, 
            device=device,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            margin=args.margin,
            neg_samples=args.neg_samples,
            save_model=args.save_model,
            model_path=args.model_path,
            patience=args.patience
        )
        
        print(f"Trainer initialized with device: {device}")
        print("Training model...")
        
        # 训练模型并获取嵌入
        embeddings = trainer.train(data, important_targets)
    
    # 计算靶点优先级
    results = []
    
    for _, row in db_data.iterrows():
        compound_id = row['compound_id']
        
        # 跳过不在图中的化合物
        if compound_id not in node_map:
            print(f"Warning: Compound {compound_id} not found in graph. Skipping...")
            continue
        
        # 计算靶点优先级（使用图结构增强）
        df = calculate_target_priorities_with_graph_structure(
            embeddings=embeddings,
            data=data,
            compound_id=compound_id,
            node_map=node_map,
            reverse_node_map=reverse_node_map,
            disease_importance=disease_importance,
            drug_similarity=drug_similarity if args.use_drug_similarity else None,
            protein_similarity=protein_similarity if args.use_protein_similarity else None,
            semantic_similarities=semantic_similarities if args.use_semantic_similarity else None,
            embedding_weight=args.embedding_weight,
            importance_weight=args.importance_weight,
            drug_sim_weight=args.drug_sim_weight if args.use_drug_similarity else 0.0,
            protein_sim_weight=args.protein_sim_weight if args.use_protein_similarity else 0.0,
            semantic_weight=args.semantic_weight if args.use_semantic_similarity else 0.0,
            structure_weight=args.structure_weight if hasattr(args, 'structure_weight') else 0.1,
            k=args.top_k
        )
        
        results.append(df)
    
    # 合并结果
    if results:
        all_results = pd.concat(results, ignore_index=True)
        
        # 保存合并结果
        os.makedirs('results/prioritization', exist_ok=True)
        all_results.to_csv('results/prioritization/target_priorities.csv', index=False)
        print(f"Combined results saved to results/prioritization/target_priorities.csv")
        
        # 创建可视化
        create_enhanced_visualization(embeddings, data, node_map, reverse_node_map, all_results)
        
        # 如果指定，则进行验证
        if args.validate and validated_data is not None:
            validate_enhanced_predictions(all_results, validated_data)
    else:
        print("Warning: No predictions generated")
    
    print("Done!")

if __name__ == "__main__":
    main()
