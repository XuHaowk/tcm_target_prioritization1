"""
验证数据生成器

该模块提供用于扩展验证交互数据的函数。
"""
import pandas as pd
import random
from tqdm import tqdm

def generate_extended_validation_data(existing_validation_path, kg_data, output_path=None, validation_ratio=0.15):
    """
    生成扩展的验证交互数据
    
    参数:
        existing_validation_path: The path to existing validation data
        kg_data: 知识图谱DataFrame
        output_path: 输出文件路径
        validation_ratio: 要选择作为验证对的化合物-靶点对的比例
        
    返回:
        扩展的验证交互DataFrame
    """
    print(f"正在从 {existing_validation_path} 加载验证数据...")
    
    # 加载现有验证数据
    validation_data = pd.read_csv(existing_validation_path)
    
    # 确定列名
    compound_col = 'compound_id' if 'compound_id' in validation_data.columns else 'compound'
    target_col = 'target_id' if 'target_id' in validation_data.columns else 'target'
    
    # 提取现有的验证对
    existing_validations = set()
    for _, row in validation_data.iterrows():
        if compound_col in row and target_col in row:
            existing_validations.add((row[compound_col], row[target_col]))
    
    print(f"找到 {len(existing_validations)} 个现有验证对")
    
    # 从KG中提取所有化合物-靶点对
    all_pairs = set()
    for _, row in kg_data.iterrows():
        all_pairs.add((row['compound'], row['target']))
    
    print(f"知识图谱中有 {len(all_pairs)} 个化合物-靶点对")
    
    # 找出潜在的新验证对
    potential_new_validations = all_pairs - existing_validations
    
    print(f"找到 {len(potential_new_validations)} 个潜在的新验证对")
    
    # 随机选择一部分作为新的验证对
    n_new_validations = int(len(potential_new_validations) * validation_ratio)
    new_validation_pairs = random.sample(list(potential_new_validations), 
                                        min(n_new_validations, len(potential_new_validations)))
    
    print(f"选择 {len(new_validation_pairs)} 个新验证对")
    
    # 创建新的验证数据
    new_validations = []
    for compound, target in tqdm(new_validation_pairs, desc="生成验证数据"):
        # 通过KG中的置信度得分来决定是否包含
        kg_entries = kg_data[(kg_data['compound'] == compound) & (kg_data['target'] == target)]
        if not kg_entries.empty and 'confidence_score' in kg_entries.columns:
            confidence = kg_entries['confidence_score'].iloc[0]
            # 仅包含高置信度的关系作为验证数据
            if confidence > 0.7 or random.random() < 0.3:  # 保留一些随机性
                new_validations.append({
                    compound_col: compound,
                    target_col: target,
                    'is_validated': 1
                })
    
    # 确保验证数据有正确的列名
    if compound_col != 'compound_id' and 'compound' in validation_data.columns:
        validation_data = validation_data.rename(columns={'compound': 'compound_id'})
    if target_col != 'target_id' and 'target' in validation_data.columns:
        validation_data = validation_data.rename(columns={'target': 'target_id'})
    
    # 合并现有和新生成的验证数据
    new_validations_df = pd.DataFrame(new_validations)
    extended_validation = pd.concat([validation_data, new_validations_df], ignore_index=True)
    
    print(f"扩展后的验证数据包含 {len(extended_validation)} 个验证交互")
    
    # 保存扩展的验证数据
    if output_path:
        extended_validation.to_csv(output_path, index=False)
        print(f"扩展的验证数据已保存到 {output_path}")
    
    return extended_validation
