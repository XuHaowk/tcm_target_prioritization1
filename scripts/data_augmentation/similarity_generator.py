"""
相似性矩阵生成器

该模块提供用于扩展药物和蛋白质相似性矩阵的函数。
"""
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

def generate_extended_similarity_matrices(kg_data, existing_drug_sim_path, existing_protein_sim_path, 
                                         drug_output_path=None, protein_output_path=None):
    """
    生成扩展的相似性矩阵
    
    参数:
        kg_data: 知识图谱DataFrame
        existing_drug_sim_path: 现有药物相似性矩阵的路径
        existing_protein_sim_path: 现有蛋白质相似性矩阵的路径
        drug_output_path: 药物相似性输出路径
        protein_output_path: 蛋白质相似性输出路径
        
    返回:
        (drug_sim, protein_sim): 扩展的药物和蛋白质相似性矩阵DataFrame
    """
    # 提取所有化合物和靶点
    all_compounds = list(kg_data['compound'].unique())
    all_targets = list(kg_data['target'].unique())
    
    print(f"知识图谱中有 {len(all_compounds)} 个化合物和 {len(all_targets)} 个靶点")
    
    drug_sim = None
    protein_sim = None
    
    # 扩展药物相似性矩阵
    try:
        print(f"正在从 {existing_drug_sim_path} 加载药物相似性矩阵...")
        drug_sim = pd.read_csv(existing_drug_sim_path, index_col=0)
        existing_compounds = drug_sim.index
        new_compounds = [c for c in all_compounds if c not in existing_compounds]
        
        print(f"找到 {len(existing_compounds)} 个现有化合物和 {len(new_compounds)} 个新化合物")
        
        if new_compounds:
            print("正在扩展药物相似性矩阵...")
            
            # 创建扩展矩阵
            new_cols = pd.DataFrame(0, index=existing_compounds, columns=new_compounds)
            new_rows = pd.DataFrame(0, index=new_compounds, columns=list(existing_compounds) + new_compounds)
            
            # 将对角线设为1
            for compound in new_compounds:
                new_rows.loc[compound, compound] = 1.0
            
            # 合并矩阵
            drug_sim = pd.concat([drug_sim, new_cols], axis=1)
            drug_sim = pd.concat([drug_sim, new_rows], axis=0)
            
            # 为新化合物生成相似性
            for new_compound in tqdm(new_compounds, desc="计算药物相似性"):
                # 查找KG中与该化合物相关的靶点
                compound_targets = kg_data[kg_data['compound'] == new_compound]['target'].unique()
                
                # 为每个现有化合物计算相似性
                for existing_compound in existing_compounds:
                    # 查找与现有化合物相关的靶点
                    existing_targets = kg_data[kg_data['compound'] == existing_compound]['target'].unique()
                    
                    # 计算共享靶点的Jaccard相似度
                    shared_targets = set(compound_targets) & set(existing_targets)
                    all_shared_targets = set(compound_targets) | set(existing_targets)
                    
                    if all_shared_targets:
                        sim = len(shared_targets) / len(all_shared_targets)
                        # 添加一些随机噪声使分布更自然
                        sim = sim * 0.7 + random.uniform(0, 0.3)
                    else:
                        sim = random.uniform(0, 0.3)  # 低基础相似度
                    
                    # 更新相似性值
                    drug_sim.loc[existing_compound, new_compound] = min(1.0, max(0.0, sim))
                    drug_sim.loc[new_compound, existing_compound] = min(1.0, max(0.0, sim))
        
        # 保存扩展的药物相似性矩阵
        if drug_output_path:
            drug_sim.to_csv(drug_output_path)
            print(f"扩展的药物相似性矩阵已保存到 {drug_output_path}")
    
    except Exception as e:
        print(f"无法扩展药物相似性矩阵: {e}")
    
    # 扩展蛋白质相似性矩阵 (类似逻辑)
    try:
        print(f"正在从 {existing_protein_sim_path} 加载蛋白质相似性矩阵...")
        protein_sim = pd.read_csv(existing_protein_sim_path, index_col=0)
        existing_targets = protein_sim.index
        new_targets = [t for t in all_targets if t not in existing_targets]
        
        print(f"找到 {len(existing_targets)} 个现有靶点和 {len(new_targets)} 个新靶点")
        
        if new_targets:
            print("正在扩展蛋白质相似性矩阵...")
            
            # 创建扩展矩阵
            new_cols = pd.DataFrame(0, index=existing_targets, columns=new_targets)
            new_rows = pd.DataFrame(0, index=new_targets, columns=list(existing_targets) + new_targets)
            
            # 将对角线设为1
            for target in new_targets:
                new_rows.loc[target, target] = 1.0
            
            # 合并矩阵
            protein_sim = pd.concat([protein_sim, new_cols], axis=1)
            protein_sim = pd.concat([protein_sim, new_rows], axis=0)
            
            # 为新靶点生成相似性
            for new_target in tqdm(new_targets, desc="计算蛋白质相似性"):
                # 查找KG中与该靶点相关的化合物
                target_compounds = kg_data[kg_data['target'] == new_target]['compound'].unique()
                
                # 为每个现有靶点计算相似性
                for existing_target in existing_targets:
                    # 查找与现有靶点相关的化合物
                    existing_compounds = kg_data[kg_data['target'] == existing_target]['compound'].unique()
                    
                    # 计算共享化合物的Jaccard相似度
                    shared_compounds = set(target_compounds) & set(existing_compounds)
                    all_compounds_set = set(target_compounds) | set(existing_compounds)
                    
                    if all_compounds_set:
                        sim = len(shared_compounds) / len(all_compounds_set)
                        # 添加一些随机噪声使分布更自然
                        sim = sim * 0.7 + random.uniform(0, 0.3)
                    else:
                        sim = random.uniform(0, 0.3)  # 低基础相似度
                    
                    # 更新相似性值
                    protein_sim.loc[existing_target, new_target] = min(1.0, max(0.0, sim))
                    protein_sim.loc[new_target, existing_target] = min(1.0, max(0.0, sim))
        
        # 保存扩展的蛋白质相似性矩阵
        if protein_output_path:
            protein_sim.to_csv(protein_output_path)
            print(f"扩展的蛋白质相似性矩阵已保存到 {protein_output_path}")
    
    except Exception as e:
        print(f"无法扩展蛋白质相似性矩阵: {e}")
    
    return drug_sim, protein_sim
