"""
知识图谱数据生成器

该模块提供用于扩展知识图谱数据的函数，增加化合物-靶点关系。
"""
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

def generate_extended_kg_data(existing_kg_path, compound_structures_path=None, output_path=None, n_new_relations=200):
    """
    生成扩展的知识图谱数据
    
    参数:
        existing_kg_path: 现有知识图谱数据的路径
        compound_structures_path: 化合物结构数据的路径 (可选)
        output_path: 输出文件路径
        n_new_relations: 要生成的新关系数量
        
    返回:
        扩展的知识图谱DataFrame
    """
    print(f"正在从 {existing_kg_path} 加载知识图谱数据...")
    
    # 加载现有数据
    kg_data = pd.read_csv(existing_kg_path)
    
    # 提取现有的化合物和靶点
    compounds = kg_data['compound'].unique()
    targets = kg_data['target'].unique()
    
    print(f"找到 {len(compounds)} 个化合物和 {len(targets)} 个靶点")
    
    # 定义可能的关系类型和权重
    relation_types = ['activation', 'inhibition', 'binding', 'modulation', 
                     'substrate', 'transport', 'indirect_regulation']
    relation_weights = [0.3, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02]  # 不同关系类型的概率
    
    # 为现有的化合物-靶点对加载已知的关联
    existing_pairs = set()
    for _, row in kg_data.iterrows():
        existing_pairs.add((row['compound'], row['target']))
    
    print(f"找到 {len(existing_pairs)} 个现有的化合物-靶点关系")
    
    # 创建新的关系
    new_relations = []
    added_pairs = set()
    
    # 方法1: 基于现有化合物和靶点创建新关系
    for _ in tqdm(range(n_new_relations // 2), desc="生成新的化合物-靶点关系"):
        compound = random.choice(compounds)
        target = random.choice(targets)
        relation_type = random.choices(relation_types, weights=relation_weights, k=1)[0]
        confidence = round(random.uniform(0.6, 0.95), 2)
        
        # 检查是否已存在此关系
        if (compound, target) not in existing_pairs and (compound, target) not in added_pairs:
            new_relations.append({
                'compound': compound,
                'target': target,
                'relation_type': relation_type,
                'confidence_score': confidence
            })
            added_pairs.add((compound, target))
    
    # 方法2: 创建新的化合物变体
    if compound_structures_path:
        try:
            print(f"正在从 {compound_structures_path} 加载化合物结构...")
            structures = pd.read_csv(compound_structures_path)
            # 获取SMILES并创建化合物变体
            for _, row in tqdm(structures.iterrows(), total=len(structures), desc="创建化合物变体"):
                if 'smiles' in row and pd.notna(row['smiles']):
                    try:
                        # 创建分子变体
                        compound_id_col = 'compound_id' if 'compound_id' in row else 'compound'
                        new_compound_name = f"{row[compound_id_col]}_variant_{len(new_relations) % 100}"
                        
                        # 为新化合物创建与某些靶点的关系
                        for _ in range(3):  # 每个新化合物添加3个关系
                            target = random.choice(targets)
                            relation_type = random.choices(relation_types, weights=relation_weights, k=1)[0]
                            confidence = round(random.uniform(0.5, 0.9), 2)
                            
                            new_relations.append({
                                'compound': new_compound_name,
                                'target': target,
                                'relation_type': relation_type,
                                'confidence_score': confidence
                            })
                    except Exception as e:
                        continue
        except Exception as e:
            print(f"无法处理化合物结构文件: {e}")
    
    # 合并现有和新生成的关系
    extended_kg = pd.concat([kg_data, pd.DataFrame(new_relations)], ignore_index=True)
    
    print(f"成功生成 {len(new_relations)} 个新关系")
    print(f"扩展后的知识图谱包含 {len(extended_kg)} 个关系")
    
    # 保存扩展的知识图谱
    if output_path:
        extended_kg.to_csv(output_path, index=False)
        print(f"扩展的知识图谱已保存到 {output_path}")
    
    return extended_kg
