"""
增强的评估指标

这个模块提供了用于评估模型性能的增强指标计算功能。
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, roc_auc_score

def calculate_enhanced_metrics(predictions, ground_truth, k_values=[1, 3, 5, 10]):
    """
    计算评估指标
    
    参数:
        predictions: 带预测的DataFrame
        ground_truth: 带验证交互的DataFrame
        k_values: Hit@k的k值列表
        
    返回:
        包含评估指标的字典
    """
    # 标准化列名
    if 'compound_id' not in ground_truth.columns and 'compound' in ground_truth.columns:
        ground_truth = ground_truth.rename(columns={'compound': 'compound_id'})
    
    if 'target_id' not in ground_truth.columns and 'target' in ground_truth.columns:
        ground_truth = ground_truth.rename(columns={'target': 'target_id'})
    
    # 提取预测的化合物-靶点对
    predicted_pairs = {}
    compound_targets = {}
    
    for _, row in predictions.iterrows():
        compound_id = row['compound_id']
        target_id = row['target_id']
        score = row['final_score']
        
        if compound_id not in predicted_pairs:
            predicted_pairs[compound_id] = []
            compound_targets[compound_id] = []
        
        predicted_pairs[compound_id].append((target_id, score))
        compound_targets[compound_id].append(target_id)
    
    # 按得分排序
    for compound_id in predicted_pairs:
        predicted_pairs[compound_id].sort(key=lambda x: x[1], reverse=True)
    
    # 提取已验证的化合物-靶点对
    validated_pairs = set()
    validated_compounds = set()
    validated_targets = set()
    
    for _, row in ground_truth.iterrows():
        compound_id = row['compound_id']
        target_id = row['target_id']
        
        if pd.notna(compound_id) and pd.notna(target_id):
            validated_pairs.add((compound_id, target_id))
            validated_compounds.add(compound_id)
            validated_targets.add(target_id)
    
    # 计算Hit@k
    hits_at_k = {k: 0 for k in k_values}
    predictions_at_k = {k: 0 for k in k_values}
    
    for compound_id, target_scores in predicted_pairs.items():
        # 跳过未验证的化合物
        if compound_id not in validated_compounds:
            continue
        
        # 对每个k统计预测
        for k in k_values:
            predictions_at_k[k] += 1
            top_k_targets = [target for target, _ in target_scores[:k]]
            
            # 检查是否有已验证的靶点在前k个预测中
            for target_id in top_k_targets:
                if (compound_id, target_id) in validated_pairs:
                    hits_at_k[k] += 1
                    break
    
    # 计算Hit@k比率
    hit_at_k_ratios = {f"hit@{k}": hits_at_k[k] / predictions_at_k[k] if predictions_at_k[k] > 0 else 0.0 
                       for k in k_values}
    
    # 计算平均倒数排名（MRR）
    mrr_sum = 0.0
    mrr_count = 0
    
    for compound_id, target_scores in predicted_pairs.items():
        # 跳过未验证的化合物
        if compound_id not in validated_compounds:
            continue
        
        mrr_count += 1
        
        # 获取第一个已验证靶点的排名
        for i, (target_id, _) in enumerate(target_scores):
            if (compound_id, target_id) in validated_pairs:
                mrr_sum += 1.0 / (i + 1)
                break
    
    # 计算平均MRR
    mrr = mrr_sum / mrr_count if mrr_count > 0 else 0.0
    
    # 计算精确率-召回率曲线和AUC
    y_true = []
    y_scores = []
    
    for compound_id, target_scores in predicted_pairs.items():
        for target_id, score in target_scores:
            is_validated = 1 if (compound_id, target_id) in validated_pairs else 0
            y_true.append(is_validated)
            y_scores.append(score)
    
    # 跳过如果没有预测或所有为真/假
    if not y_true or len(set(y_true)) < 2:
        pr_auc = np.nan
        roc_auc = np.nan
        precision = []
        recall = []
        fpr = []
        tpr = []
    else:
        # 计算精确率-召回率曲线
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
    
    # 计算按排名的召回率
    recall_at_rank = []
    ranks = [1, 5, 10, 20, 50, 100]
    actual_positives = sum(y_true)
    
    for rank in ranks:
        if len(y_scores) >= rank:
            # 按得分排序
            paired_scores = list(zip(y_scores, y_true))
            paired_scores.sort(key=lambda x: x[0], reverse=True)
            
            # 计算前k个中的正例数量
            positives_at_k = sum(truth for _, truth in paired_scores[:rank])
            recall = positives_at_k / actual_positives if actual_positives > 0 else 0
        else:
            recall = np.nan
            
        recall_at_rank.append((rank, recall))
    
    # 编译所有指标
    metrics = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "mrr": mrr,
        **hit_at_k_ratios,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "tpr": tpr,
        "recall_at_rank": recall_at_rank
    }
    
    return metrics

def plot_enhanced_validation_metrics(metrics, save_path=None):
    """
    绘制验证指标
    
    参数:
        metrics: 评估指标字典
        save_path: 保存路径
        
    返回:
        Figure对象
    """
    # 创建2行2列的图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 设置图标题
    fig.suptitle('Validation Metrics', fontsize=16)
    
    # 绘制精确率-召回率曲线
    if isinstance(metrics.get("precision", []), np.ndarray) and len(metrics.get("precision", [])) > 0:
        axes[0, 0].plot(metrics["recall"], metrics["precision"], label=f'PR AUC = {metrics["pr_auc"]:.3f}')
        axes[0, 0].set_title('Precision-Recall Curve')
        axes[0, 0].set_xlabel('Recall')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    else:
        axes[0, 0].text(0.5, 0.5, 'Insufficient data for PR curve', 
                        horizontalalignment='center', verticalalignment='center')
    
    # 绘制ROC曲线
    if isinstance(metrics.get("fpr", []), np.ndarray) and len(metrics.get("fpr", [])) > 0:
        axes[0, 1].plot(metrics["fpr"], metrics["tpr"], label=f'ROC AUC = {metrics["roc_auc"]:.3f}')
        axes[0, 1].plot([0, 1], [0, 1], 'k--')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    else:
        axes[0, 1].text(0.5, 0.5, 'Insufficient data for ROC curve', 
                        horizontalalignment='center', verticalalignment='center')
    
    # 绘制Hit@k
    hit_k_metrics = {k: v for k, v in metrics.items() if k.startswith("hit@")}
    if hit_k_metrics:
        k_values = [int(k.split("@")[1]) for k in hit_k_metrics.keys()]
        hit_rates = [hit_k_metrics[f"hit@{k}"] for k in k_values]
        
        axes[1, 0].bar(k_values, hit_rates, color='steelblue')
        axes[1, 0].set_title('Hit@k')
        axes[1, 0].set_xlabel('k')
        axes[1, 0].set_ylabel('Hit Rate')
        axes[1, 0].set_xticks(k_values)
        axes[1, 0].grid(True, axis='y')
        
        # 添加数值标签
        for i, v in enumerate(hit_rates):
            axes[1, 0].text(k_values[i], v + 0.02, f'{v:.2f}', 
                            ha='center', va='bottom', fontweight='bold')
    else:
        axes[1, 0].text(0.5, 0.5, 'No Hit@k metrics available', 
                        horizontalalignment='center', verticalalignment='center')
    
    # 绘制MRR和按排名的召回率
    if "mrr" in metrics:
        ax_twin = axes[1, 1].twinx()
        
        # 绘制MRR
        axes[1, 1].bar(['MRR'], [metrics["mrr"]], color='darkblue', alpha=0.7)
        axes[1, 1].set_ylabel('MRR', color='darkblue')
        axes[1, 1].tick_params(axis='y', labelcolor='darkblue')
        axes[1, 1].text(0, metrics["mrr"] + 0.02, f'{metrics["mrr"]:.3f}', 
                        ha='center', va='bottom', fontweight='bold', color='darkblue')
        
        # 绘制按排名的召回率
        if "recall_at_rank" in metrics and metrics["recall_at_rank"]:
            ranks, recalls = zip(*metrics["recall_at_rank"])
            ax_twin.plot(list(map(str, ranks)), recalls, 'o-', color='darkorange', label='Recall@Rank')
            ax_twin.set_ylabel('Recall', color='darkorange')
            ax_twin.tick_params(axis='y', labelcolor='darkorange')
            
            # 添加图例
            ax_twin.legend(loc='upper right')
        
        axes[1, 1].set_title('Mean Reciprocal Rank & Recall@Rank')
    else:
        axes[1, 1].text(0.5, 0.5, 'No MRR metric available', 
                        horizontalalignment='center', verticalalignment='center')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Validation metrics plot saved to {save_path}")
    
    return fig
