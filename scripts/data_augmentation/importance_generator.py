"""
疾病重要性数据生成器

该模块提供用于扩展疾病重要性数据的函数，为新靶点添加重要性得分。
"""
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

def generate_extended_importance_data(existing_importance_path, kg_data, output_path=None):
    """
    生成扩展的疾病重要性数据
    
    参数:
        existing_importance_path: 现有疾病重要性数据的路径
        kg_data: 知识图谱DataFrame
        output_path: 输出文件路径
        
    返回:
        扩展的疾病重要性DataFrame
    """
    print(f"正在从 {existing_importance_path} 加载疾病重要性数据...")
    
    # 加载现有数据
    importance_data = pd.read_csv(existing_importance_path)
    
    # 查找靶点列名
    target_column = None
    possible_target_columns = ['target_id', 'target', 'protein_id', 'protein', 'gene_id', 'gene']
    
    for col in possible_target_columns:
        if col in importance_data.columns:
            target_column = col
            break
    
    if target_column is None:
        target_column = importance_data.columns[0]
        print(f"警告: 无法找到靶点列名，使用第一列 '{target_column}'")
    
    # 查找重要性列名
    importance_column = None
    possible_importance_columns = ['importance_score', 'importance', 'score', 'weight', 'priority']
    
    for col in possible_importance_columns:
        if col in importance_data.columns:
            importance_column = col
            break
    
    if importance_column is None:
        importance_column = importance_data.columns[1] if len(importance_data.columns) > 1 else None
        if importance_column:
            print(f"警告: 无法找到重要性列名，使用 '{importance_column}'")
        else:
            print(f"错误: 找不到合适的重要性列名")
            return importance_data
    
    # 提取现有的靶点
    existing_targets = set(importance_data[target_column].values)
    print(f"找到 {len(existing_targets)} 个现有靶点")
    
    # 从KG中提取所有靶点
    all_targets = set(kg_data['target'].unique())
    print(f"知识图谱中有 {len(all_targets)} 个靶点")
    
    # 找出新靶点
    new_targets = all_targets - existing_targets
    print(f"找到 {len(new_targets)} 个新靶点")
    
    # 为新靶点创建重要性得分
    new_importance = []
    
    for target in tqdm(new_targets, desc="生成靶点重要性"):
        # 分析该靶点在KG中的连接性
        target_relations = kg_data[kg_data['target'] == target]
        
        # 根据关系数量和类型设置基础重要性
        n_relations = len(target_relations)
        base_importance = 0.1 + 0.2 * np.log1p(n_relations) / np.log1p(10)
        
        # 根据关系类型调整重要性
        relation_type_factor = 1.0
        if 'relation_type' in target_relations.columns:
            # 对激活和抑制关系给予更高权重
            activation_count = sum(1 for rt in target_relations['relation_type'] if isinstance(rt, str) and 'activ' in rt.lower())
            inhibition_count = sum(1 for rt in target_relations['relation_type'] if isinstance(rt, str) and 'inhib' in rt.lower())
            relation_type_factor += 0.15 * (activation_count + inhibition_count) / max(1, n_relations)
        
        # 计算最终重要性并加入随机性
        importance = base_importance * relation_type_factor
        importance = min(0.95, max(0.05, importance + random.uniform(-0.1, 0.1)))
        
        new_importance_entry = {target_column: target, importance_column: round(importance, 3)}
        new_importance.append(new_importance_entry)
    
    # 合并现有和新生成的重要性数据
    extended_importance = pd.concat([importance_data, pd.DataFrame(new_importance)], ignore_index=True)
    
    print(f"扩展后的疾病重要性数据包含 {len(extended_importance)} 个靶点")
    
    # 保存扩展的重要性数据
    if output_path:
        extended_importance.to_csv(output_path, index=False)
        print(f"扩展的疾病重要性数据已保存到 {output_path}")
    
    return extended_importance
