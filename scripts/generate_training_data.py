#!/usr/bin/env python3
"""
矽肺靶点优先级排序系统 - 训练数据生成器

该脚本生成扩展的训练数据，包括知识图谱、疾病重要性、相似性矩阵和验证数据。
"""
import os
import argparse
from tqdm import tqdm

from data_augmentation import (
    generate_extended_kg_data,
    generate_extended_importance_data,
    generate_extended_similarity_matrices,
    generate_extended_validation_data,
    generate_semantic_similarities
)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='生成扩展的训练数据')
    
    parser.add_argument('--input_dir', type=str, default='data/raw',
                        help='输入数据目录')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                        help='处理后数据目录')
    parser.add_argument('--output_dir', type=str, default='data/extended',
                        help='输出数据目录')
    parser.add_argument('--kg_file', type=str, default='kg_data_extended.csv',
                        help='知识图谱文件名')
    parser.add_argument('--structures_file', type=str, default='drug_structures.csv',
                        help='化合物结构文件名')
    parser.add_argument('--importance_file', type=str, default='disease_importance_extended.csv',
                        help='疾病重要性文件名')
    parser.add_argument('--validation_file', type=str, default='validated_interactions.csv',
                        help='验证交互文件名')
    parser.add_argument('--n_relations', type=int, default=300,
                        help='要生成的新关系数量')
    parser.add_argument('--validation_ratio', type=float, default=0.2,
                        help='要用作验证数据的关系比例')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 定义输入和输出路径
    kg_path = os.path.join(args.input_dir, args.kg_file)
    structures_path = os.path.join(args.input_dir, args.structures_file)
    importance_path = os.path.join(args.input_dir, args.importance_file)
    validation_path = os.path.join(args.input_dir, args.validation_file)
    
    drug_sim_path = os.path.join(args.processed_dir, 'drug_similarity.csv')
    protein_sim_path = os.path.join(args.processed_dir, 'protein_similarity.csv')
    
    extended_kg_path = os.path.join(args.output_dir, 'kg_data_extended_large.csv')
    extended_importance_path = os.path.join(args.output_dir, 'disease_importance_extended_large.csv')
    extended_validation_path = os.path.join(args.output_dir, 'validated_interactions_extended.csv')
    extended_drug_sim_path = os.path.join(args.output_dir, 'drug_similarity_extended.csv')
    extended_protein_sim_path = os.path.join(args.output_dir, 'protein_similarity_extended.csv')
    semantic_sim_path = os.path.join(args.output_dir, 'semantic_similarities_extended.csv')
    
    print("\n" + "="*80)
    print("矽肺靶点优先级排序系统 - 训练数据生成器")
    print("="*80 + "\n")
    
    # 检查输入文件
    missing_files = []
    for file_path in [kg_path, importance_path, validation_path]:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("错误: 找不到以下输入文件:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return
    
    # 1. 生成扩展的知识图谱数据
    print("\n" + "-"*80)
    print("1. 生成扩展的知识图谱数据")
    print("-"*80)
    
    structures_path_to_use = structures_path if os.path.exists(structures_path) else None
    if not structures_path_to_use:
        print(f"注意: 找不到化合物结构文件 {structures_path}，将跳过基于结构的扩展")
    
    extended_kg = generate_extended_kg_data(
        kg_path, structures_path_to_use, extended_kg_path, n_new_relations=args.n_relations
    )
    
    # 2. 生成扩展的疾病重要性数据
    print("\n" + "-"*80)
    print("2. 生成扩展的疾病重要性数据")
    print("-"*80)
    
    extended_importance = generate_extended_importance_data(
        importance_path, extended_kg, extended_importance_path
    )
    
    # 3. 生成扩展的验证交互数据
    print("\n" + "-"*80)
    print("3. 生成扩展的验证交互数据")
    print("-"*80)
    
    extended_validation = generate_extended_validation_data(
        validation_path, extended_kg, extended_validation_path, validation_ratio=args.validation_ratio
    )
    
    # 4. 生成扩展的相似性矩阵
    print("\n" + "-"*80)
    print("4. 生成扩展的相似性矩阵")
    print("-"*80)
    
    drug_sim_path_to_use = drug_sim_path if os.path.exists(drug_sim_path) else None
    protein_sim_path_to_use = protein_sim_path if os.path.exists(protein_sim_path) else None
    
    if not drug_sim_path_to_use:
        print(f"注意: 找不到药物相似性矩阵 {drug_sim_path}，将跳过药物相似性扩展")
    if not protein_sim_path_to_use:
        print(f"注意: 找不到蛋白质相似性矩阵 {protein_sim_path}，将跳过蛋白质相似性扩展")
    
    if drug_sim_path_to_use or protein_sim_path_to_use:
        drug_sim, protein_sim = generate_extended_similarity_matrices(
            extended_kg, drug_sim_path_to_use, protein_sim_path_to_use,
            extended_drug_sim_path, extended_protein_sim_path
        )
    
    # 5. 生成语义相似性数据
    print("\n" + "-"*80)
    print("5. 生成语义相似性数据")
    print("-"*80)
    
    semantic_similarities = generate_semantic_similarities(
        extended_kg, semantic_sim_path
    )
    
    # 总结
    print("\n" + "="*80)
    print("数据生成完成!")
    print("="*80)
    
    print(f"\n扩展的数据文件位于: {args.output_dir}")
    print(f"- 知识图谱数据: {len(extended_kg)} 条关系")
    print(f"- 疾病重要性数据: {len(extended_importance)} 个靶点")
    print(f"- 验证交互数据: {len(extended_validation)} 个验证对")
    print(f"- 语义相似性数据: {len(semantic_similarities)} 个相似度条目")
    
    # 打印下一步命令
    print("\n要使用扩展的数据集训练模型，请运行:")
    print(f"bash scripts/run_with_extended_data.sh")

if __name__ == "__main__":
    main()
