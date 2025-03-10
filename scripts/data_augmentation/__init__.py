"""
矽肺靶点优先级排序系统 - 数据扩充模块

该模块提供用于生成扩展训练数据的工具，以提高模型性能。
包含知识图谱、疾病重要性、相似性矩阵和验证数据的生成函数。
"""

from .kg_generator import generate_extended_kg_data
from .importance_generator import generate_extended_importance_data
from .similarity_generator import generate_extended_similarity_matrices
from .validation_generator import generate_extended_validation_data
from .semantic_generator import generate_semantic_similarities
