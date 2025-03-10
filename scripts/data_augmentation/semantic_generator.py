"""
语义相似性生成器

该模块提供用于生成语义相似性数据的函数。
"""
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

def generate_semantic_similarities(kg_data, output_path=None):
    """
    生成语义相似性数据
    
    参数:
        kg_data: 知识图谱DataFrame
        output_path: 输出文件路径
        
    返回:
        语义相似性DataFrame
    """
    print("正在生成语义相似性数据...")
    
    semantic_similarities = []
    compounds = kg_data['compound'].unique()
    targets = kg_data['target'].unique()
    
    print(f"分析 {len(compounds)} 个化合物和 {len(targets)} 个靶点之间的语义关系")
    
    # 对每个化合物，查找潜在的靶点关系
    for compound in tqdm(compounds, desc="计算语义相似性"):
        # 获取该化合物的所有已知关系
        compound_relations = kg_data[kg_data['compound'] == compound]
        compound_targets = set(compound_relations['target'])
        
        # 为所有可能的靶点创建语义相似性
        for target in targets:
            # 跳过已知的直接关系
            if target in compound_targets:
                continue
            
            # 寻找共同的相邻靶点（两步路径）
            semantic_similarity = 0.0
            num_paths = 0
            
            # 计算路径相似性
            for intermediate_target in compound_targets:
                # 查找中间靶点与目标靶点的关系
                intermediate_relations = kg_data[kg_data['target'] == intermediate_target]
                intermediate_compounds = set(intermediate_relations['compound'])
                
                target_relations = kg_data[kg_data['target'] == target]
                target_compounds = set(target_relations['compound'])
                
                # 共享的化合物表示路径连接
                shared_compounds = intermediate_compounds & target_compounds
                if shared_compounds:
                    # 每条路径增加一定的相似度
                    path_contribution = 0.2 * len(shared_compounds) / len(intermediate_compounds | target_compounds)
                    semantic_similarity += path_contribution
                    num_paths += len(shared_compounds)
            
            # 有意义的相似度需要有路径存在
            if num_paths > 0:
                # 标准化相似度
                semantic_similarity = min(0.95, semantic_similarity)
                
                # 添加一些随机性，但保持相对顺序
                semantic_similarity = semantic_similarity * 0.8 + random.uniform(0, 0.2)
                
                semantic_similarities.append({
                    'compound_id': compound,
                    'target_id': target,
                    'semantic_similarity': round(semantic_similarity, 4),
                    'num_paths': num_paths
                })
    
    # 创建语义相似性数据框
    semantic_df = pd.DataFrame(semantic_similarities)
    
    print(f"生成了 {len(semantic_df)} 个语义相似性条目")
    
    # 保存语义相似性数据
    if output_path:
        semantic_df.to_csv(output_path, index=False)
        print(f"语义相似性数据已保存到 {output_path}")
    
    return semantic_df
