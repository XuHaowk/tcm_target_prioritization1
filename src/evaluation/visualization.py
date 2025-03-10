"""
增强的可视化模块

这个模块提供了用于可视化嵌入和预测结果的增强功能。
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def create_enhanced_visualization(embeddings, data, node_map, reverse_node_map, predictions):
    """
    创建增强的可视化
    
    参数:
        embeddings: 节点嵌入
        data: PyG数据对象
        node_map: 节点ID到索引的映射
        reverse_node_map: 索引到节点ID的映射
        predictions: 预测结果DataFrame
    """
    # 创建输出目录
    os.makedirs('results/visualizations', exist_ok=True)
    
    # 转换嵌入为NumPy数组
    embeddings_np = embeddings.detach().cpu().numpy()
    
    # 提取化合物和靶点索引
    compound_indices = data.compound_indices.cpu().numpy()
    target_indices = data.target_indices.cpu().numpy()
    
    # 获取嵌入标签
    labels = []
    node_ids = []
    
    for i in range(embeddings_np.shape[0]):
        node_id = reverse_node_map[i]
        node_ids.append(node_id)
        
        if i in compound_indices:
            labels.append('Compound')
        elif i in target_indices:
            labels.append('Target')
        else:
            labels.append('Unknown')
    
    # 应用t-SNE降维
    try:
        # 确定适当的困惑度参数
        n_samples = embeddings_np.shape[0]
        perplexity = min(n_samples - 1, 15)  # 使用较小的困惑度值
        
        print(f"Applying t-SNE with perplexity={perplexity} for {n_samples} nodes...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings_np)
        
        # 创建用于绘图的DataFrame
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'label': labels,
            'node_id': node_ids
        })
        
        # 绘制嵌入
        plt.figure(figsize=(12, 10))
        
        # 绘制化合物
        compounds = df[df['label'] == 'Compound']
        plt.scatter(compounds['x'], compounds['y'], c='royalblue', marker='o', s=100, alpha=0.8, label='Compound')
        
        # 绘制靶点
        targets = df[df['label'] == 'Target']
        plt.scatter(targets['x'], targets['y'], c='firebrick', marker='^', s=120, alpha=0.8, label='Target')
        
        # 添加节点ID作为标签
        for i, row in df.iterrows():
            label_color = 'navy' if row['label'] == 'Compound' else 'darkred'
            plt.annotate(row['node_id'], (row['x'], row['y']), 
                        fontsize=8, alpha=0.9, color=label_color,
                        bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='gray', alpha=0.8))
        
        plt.title('t-SNE Visualization of Node Embeddings', fontsize=16)
        plt.legend(fontsize=12, markerscale=1.2)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加边框和背景
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().set_facecolor('#f8f8f8')
        
        # 保存图表
        plt.savefig('results/visualizations/embeddings_tsne.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Embeddings visualization saved to results/visualizations/embeddings_tsne.png")
        
        # 创建3D嵌入可视化
        from mpl_toolkits.mplot3d import Axes3D
        
        if embeddings_np.shape[1] >= 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # 使用前三个维度作为3D坐标
            compounds_indices = [i for i, label in enumerate(labels) if label == 'Compound']
            targets_indices = [i for i, label in enumerate(labels) if label == 'Target']
            
            # 绘制化合物
            ax.scatter(embeddings_np[compounds_indices, 0], 
                      embeddings_np[compounds_indices, 1],
                      embeddings_np[compounds_indices, 2],
                      c='royalblue', marker='o', s=100, alpha=0.8, label='Compound')
            
            # 绘制靶点
            ax.scatter(embeddings_np[targets_indices, 0], 
                      embeddings_np[targets_indices, 1],
                      embeddings_np[targets_indices, 2],
                      c='firebrick', marker='^', s=120, alpha=0.8, label='Target')
            
            ax.set_title('3D Visualization of Node Embeddings', fontsize=16)
            ax.legend(fontsize=12)
            
            # 保存图表
            plt.savefig('results/visualizations/embeddings_3d.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("3D embeddings visualization saved to results/visualizations/embeddings_3d.png")
    
    except Exception as e:
        print(f"Warning: t-SNE visualization failed: {str(e)}")
        print("Skipping t-SNE visualization and continuing...")
    
    # 创建热图
    try:
        top_compounds = predictions['compound_id'].unique()[:5]  # 取前5个化合物
        
        if len(top_compounds) == 0:
            print("No compounds in predictions, skipping heatmap")
            return
        
        # 过滤前5个化合物的预测
        filtered_predictions = predictions[predictions['compound_id'].isin(top_compounds)]
        
        # 获取所有化合物的前10个靶点
        top_targets = set()
        for compound in top_compounds:
            compound_preds = filtered_predictions[filtered_predictions['compound_id'] == compound]
            compound_preds = compound_preds.sort_values(by='final_score', ascending=False)
            top_compound_targets = compound_preds.head(10)['target_id'].tolist()
            top_targets.update(top_compound_targets)
        
        top_targets = list(top_targets)
        
        if len(top_targets) == 0:
            print("No targets in filtered predictions, skipping heatmap")
            return
        
        # 创建热图矩阵
        heatmap_data = np.zeros((len(top_compounds), len(top_targets)))
        
        # 创建得分分解热图矩阵
        component_names = ['embedding_similarity', 'importance_score', 
                           'drug_similarity_score', 'protein_similarity_score',
                           'semantic_similarity_score']
        
        # 获取预测值
        for i, compound in enumerate(top_compounds):
            for j, target in enumerate(top_targets):
                mask = (filtered_predictions['compound_id'] == compound) & \
                       (filtered_predictions['target_id'] == target)
                
                if mask.any():
                    heatmap_data[i, j] = filtered_predictions.loc[mask, 'final_score'].values[0]
        
        # 创建热图
        plt.figure(figsize=(14, 10))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu",
                   xticklabels=top_targets, yticklabels=top_compounds)
        plt.title('Top Compound-Target Prioritization Scores', fontsize=16)
        plt.xlabel('Target IDs', fontsize=14)
        plt.ylabel('Compound IDs', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # 保存热图
        plt.savefig('results/visualizations/priority_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Priority heatmap saved to results/visualizations/priority_heatmap.png")
        
        # 创建得分分解堆叠条形图
        plt.figure(figsize=(15, 10))
        
        # 获取每个化合物的前3个靶点
        for i, compound in enumerate(top_compounds):
            compound_preds = filtered_predictions[filtered_predictions['compound_id'] == compound]
            compound_preds = compound_preds.sort_values(by='final_score', ascending=False)
            top3_targets = compound_preds.head(3)
            
            if len(top3_targets) > 0:
                plt.subplot(len(top_compounds), 1, i+1)
                
                # 提取各组分得分
                components = []
                for target_row in top3_targets.itertuples():
                    target_components = []
                    for comp in component_names:
                        if hasattr(target_row, comp):
                            target_components.append(getattr(target_row, comp))
                        else:
                            target_components.append(0)
                    components.append(target_components)
                
                components = np.array(components)
                
                # 创建堆叠条形图
                bar_width = 0.6
                target_ids = top3_targets['target_id'].tolist()
                bar_positions = np.arange(len(target_ids))
                
                bottom = np.zeros(len(target_ids))
                for j, comp_name in enumerate(component_names):
                    plt.bar(bar_positions, components[:, j], bar_width, 
                           bottom=bottom, label=comp_name if i == 0 else "")
                    bottom += components[:, j]
                
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.title(f'Compound: {compound}', fontsize=12)
                plt.xticks(bar_positions, target_ids, rotation=30, ha='right')
                plt.ylabel('Score Component', fontsize=10)
                
                # 只在最后一个子图上显示x轴标签
                if i == len(top_compounds) - 1:
                    plt.xlabel('Target IDs', fontsize=12)
                
                # 只在第一个子图上显示图例
                if i == 0:
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('results/visualizations/score_components.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Score component visualization saved to results/visualizations/score_components.png")
        
    except Exception as e:
        print(f"Warning: Heatmap visualization failed: {str(e)}")
        print("Skipping heatmap visualization and continuing...")
