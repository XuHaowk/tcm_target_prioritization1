#!/usr/bin/env python3
"""
结果分析脚本

这个脚本分析不同模型架构的结果，生成比较报告。
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_metrics_file(model_name):
    """
    加载模型的指标文件
    
    参数:
        model_name: 模型名称
        
    返回:
        指标DataFrame
    """
    metrics_path = f"results/comparisons/{model_name}_metrics.csv"
    if os.path.exists(metrics_path):
        return pd.read_csv(metrics_path)
    else:
        # 尝试从日志中提取指标
        log_path = f"results/comparisons/{model_name}_log.txt"
        if os.path.exists(log_path):
            metrics = {'Metric': [], 'Value': []}
            with open(log_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'AUC:' in line or 'MRR:' in line or 'Hit@' in line:
                        parts = line.strip().split(':')
                        if len(parts) == 2:
                            metric = parts[0].strip()
                            try:
                                value = float(parts[1].strip())
                                metrics['Metric'].append(metric)
                                metrics['Value'].append(value)
                            except ValueError:
                                pass
            
            if metrics['Metric']:
                return pd.DataFrame(metrics)
    
    return None

def compare_models():
    """比较不同模型的性能"""
    model_names = ['rgcn', 'attention_rgcn', 'edge_feature_rgcn', 'multiscale_rgcn', 'graph_pool_rgcn']
    model_display_names = {
        'rgcn': 'Base RGCN',
        'attention_rgcn': 'Attention RGCN',
        'edge_feature_rgcn': 'Edge Feature RGCN',
        'multiscale_rgcn': 'Multiscale RGCN',
        'graph_pool_rgcn': 'Graph Pool RGCN'
    }
    
    # 收集所有模型的指标
    all_metrics = {}
    
    for model_name in model_names:
        metrics_df = load_metrics_file(model_name)
        if metrics_df is not None:
            all_metrics[model_name] = {}
            for _, row in metrics_df.iterrows():
                metric = row['Metric']
                value = row['Value']
                if isinstance(value, str) and value.lower() == 'n/a':
                    value = np.nan
                all_metrics[model_name][metric] = value
    
    # 创建比较表
    comparison_data = []
    
    key_metrics = ['PR AUC', 'ROC AUC', 'MRR', 'Hit@1', 'Hit@3', 'Hit@5', 'Hit@10']
    
    for model_name in model_names:
        if model_name in all_metrics:
            row = {'Model': model_display_names[model_name]}
            for metric in key_metrics:
                row[metric] = all_metrics[model_name].get(metric, np.nan)
            comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 保存比较表
    os.makedirs('results/comparisons', exist_ok=True)
    comparison_df.to_csv('results/comparisons/model_comparison.csv', index=False)
    print("Model comparison saved to results/comparisons/model_comparison.csv")
    
    # 生成比较图表
    plt.figure(figsize=(12, 8))
    
    # 宽格式转换为长格式用于绘图
    plot_data = pd.melt(comparison_df, id_vars=['Model'], 
                        value_vars=key_metrics,
                        var_name='Metric', value_name='Value')
    
    # 绘制条形图
    g = sns.catplot(x='Metric', y='Value', hue='Model', data=plot_data, 
                   kind='bar', height=6, aspect=1.5, palette='viridis')
    
    plt.title('Model Performance Comparison', fontsize=16)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('results/comparisons/model_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison chart saved to results/comparisons/model_comparison.png")
    
    # 找出表现最好的模型
    best_models = {}
    for metric in key_metrics:
        if metric in comparison_df.columns:
            best_idx = comparison_df[metric].idxmax()
            if pd.notna(best_idx):
                best_model = comparison_df.loc[best_idx, 'Model']
                best_value = comparison_df.loc[best_idx, metric]
                best_models[metric] = (best_model, best_value)
    
    # 打印最佳模型
    print("\n=== Best Models ===")
    for metric, (model, value) in best_models.items():
        print(f"{metric}: {model} ({value:.4f})")
    
    # 推荐最佳模型
    if best_models:
        # 基于MRR和Hit@10的加权得分选择最佳模型
        model_scores = {}
        for model_name in model_names:
            if model_name in all_metrics:
                mrr = all_metrics[model_name].get('MRR', 0)
                hit10 = all_metrics[model_name].get('Hit@10', 0)
                pr_auc = all_metrics[model_name].get('PR AUC', 0)
                roc_auc = all_metrics[model_name].get('ROC AUC', 0)
                
                # 加权综合得分
                score = 0.4 * mrr + 0.3 * hit10 + 0.15 * pr_auc + 0.15 * roc_auc
                model_scores[model_name] = score
        
        # 选择得分最高的模型
        best_model = max(model_scores.items(), key=lambda x: x[1])
        
        print(f"\nRecommended model: {model_display_names[best_model[0]]} (Score: {best_model[1]:.4f})")
        print(f"Model path: results/comparisons/{best_model[0]}_model.pt")
    
    return comparison_df

if __name__ == "__main__":
    compare_models()
