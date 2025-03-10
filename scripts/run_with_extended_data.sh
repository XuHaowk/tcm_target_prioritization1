#!/bin/bash
# 使用扩展数据运行矽肺靶点优先级排序系统

# 创建必要的目录
mkdir -p results/{embeddings,models,prioritization,semantic_analysis,evaluation,visualizations}

# 设置输入数据路径
EXTENDED_DATA_DIR="data/extended"
RAW_DATA_DIR="data/raw"

echo "正在使用扩展数据运行矽肺靶点优先级排序系统..."

# 确保扩展数据存在
if [ ! -f "${EXTENDED_DATA_DIR}/kg_data_extended_large.csv" ]; then
    echo "错误: 找不到扩展数据。请先运行 'python scripts/generate_training_data.py'"
    exit 1
fi

# 运行主模型
python scripts/main.py \
    --kg_data ${EXTENDED_DATA_DIR}/kg_data_extended_large.csv \
    --db_data ${RAW_DATA_DIR}/database_data_extended.csv \
    --disease_importance ${EXTENDED_DATA_DIR}/disease_importance_extended_large.csv \
    --validated_data ${EXTENDED_DATA_DIR}/validated_interactions_extended.csv \
    --drug_similarity ${EXTENDED_DATA_DIR}/drug_similarity_extended.csv \
    --protein_similarity ${EXTENDED_DATA_DIR}/protein_similarity_extended.csv \
    --semantic_similarities ${EXTENDED_DATA_DIR}/semantic_similarities_extended.csv \
    --use_drug_similarity \
    --use_protein_similarity \
    --use_semantic_similarity \
    --embedding_weight 0.05 \
    --importance_weight 0.75 \
    --drug_sim_weight 0.075 \
    --protein_sim_weight 0.075 \
    --semantic_weight 0.05 \
    --save_model \
    --validate

# 检查运行状态
if [ $? -eq 0 ]; then
    echo "模型训练和目标优先级排序完成!"
    echo "结果保存在 results/ 目录中"
else
    echo "运行过程中出现错误，请检查日志"
fi
