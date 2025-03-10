#!/bin/bash
# 运行不同模型架构的比较实验

# 创建结果目录
mkdir -p results/comparisons

# 基础命令
BASE_CMD="python scripts/main_enhanced.py \
    --kg_data data/extended/kg_data_extended_large.csv \
    --db_data data/raw/database_data_extended.csv \
    --disease_importance data/extended/disease_importance_extended_large.csv \
    --validated_data data/extended/validated_interactions_extended.csv \
    --drug_similarity data/extended/drug_similarity_extended.csv \
    --protein_similarity data/extended/protein_similarity_extended.csv \
    --semantic_similarities data/extended/semantic_similarities_extended.csv \
    --use_drug_similarity \
    --use_protein_similarity \
    --use_semantic_similarity \
    --embedding_weight 0.05 \
    --importance_weight 0.75 \
    --drug_sim_weight 0.075 \
    --protein_sim_weight 0.075 \
    --semantic_weight 0.05 \
    --structure_weight 0.1 \
    --epochs 150 \
    --batch_size 64 \
    --patience 15 \
    --validate \
    --save_model"

# 运行不同模型架构
echo "Running default RGCN model..."
$BASE_CMD --model_type rgcn \
    --model_path results/comparisons/rgcn_model.pt \
    > results/comparisons/rgcn_log.txt 2>&1

echo "Running attention-enhanced RGCN model..."
$BASE_CMD --model_type attention_rgcn \
    --model_path results/comparisons/attention_rgcn_model.pt \
    --use_layer_norm --use_residual \
    --hidden_dims 256,256,256 \
    > results/comparisons/attention_rgcn_log.txt 2>&1

echo "Running edge-feature RGCN model..."
$BASE_CMD --model_type edge_feature_rgcn \
    --model_path results/comparisons/edge_feature_rgcn_model.pt \
    --hidden_dims 256 \
    > results/comparisons/edge_feature_rgcn_log.txt 2>&1

echo "Running multi-scale RGCN model..."
$BASE_CMD --model_type multiscale_rgcn \
    --model_path results/comparisons/multiscale_rgcn_model.pt \
    --hidden_dims 256,256,256,256 \
    > results/comparisons/multiscale_rgcn_log.txt 2>&1

echo "Running graph-pooling RGCN model..."
$BASE_CMD --model_type graph_pool_rgcn \
    --model_path results/comparisons/graph_pool_rgcn_model.pt \
    --hidden_dims 256 \
    > results/comparisons/graph_pool_rgcn_log.txt 2>&1

echo "Model comparison completed!"
