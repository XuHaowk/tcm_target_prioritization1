"""
图池化增强的关系图卷积网络（GraphPoolRGCN）模型

这个模块提供了带图池化的RGCN实现，能够捕获全局结构信息。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_add_pool, global_mean_pool, global_max_pool

class GraphPoolRGCN(nn.Module):
    """
    带图池化的RGCN模型，能够捕获全局结构信息
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations, num_bases=None, 
                 dropout=0.2, pooling='attention'):
        """
        初始化GraphPoolRGCN模型

        参数:
            in_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            out_dim: 输出嵌入维度
            num_relations: 关系类型数量
            num_bases: 基矩阵数量
            dropout: Dropout概率
            pooling: 图池化类型 ('sum', 'mean', 'max', 'attention')
        """
        super(GraphPoolRGCN, self).__init__()
        
        # 基础RGCN层
        self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases)
        self.conv3 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases)
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # 图池化类型
        self.pooling = pooling
        
        # 如果使用注意力池化，添加注意力机制
        if pooling == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        # 用于组合节点和全局特征的MLP
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
        
        # Dropout
        self.dropout = dropout
        
        # 初始化权重
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
        for conv in [self.conv1, self.conv2, self.conv3]:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()
    
    def forward(self, x, edge_index, edge_type, batch=None, edge_weight=None):
        """
        前向传播

        参数:
            x: 节点特征
            edge_index: 边索引
            edge_type: 边类型
            batch: 批处理指示符（将节点分配到图）
            edge_weight: 边权重
            
        返回:
            节点嵌入
        """
        # 第一层RGCN
        x = self.conv1(x, edge_index, edge_type)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层RGCN
        x = self.conv2(x, edge_index, edge_type)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第三层RGCN
        node_emb = self.conv3(x, edge_index, edge_type)
        
        # 如果没有提供批处理信息，假设所有节点属于同一个图
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # 应用不同类型的图池化
        if self.pooling == 'sum':
            global_emb = global_add_pool(node_emb, batch)
        elif self.pooling == 'mean':
            global_emb = global_mean_pool(node_emb, batch)
        elif self.pooling == 'max':
            global_emb = global_max_pool(node_emb, batch)
        elif self.pooling == 'attention':
            # 计算注意力得分
            scores = self.attention(node_emb)
            attention_weights = torch.softmax(scores, dim=0)
            # 加权聚合
            global_emb = torch.zeros((batch.max() + 1, node_emb.size(1)), device=node_emb.device)
            for i in range(batch.max() + 1):
                mask = (batch == i)
                global_emb[i] = torch.sum(node_emb[mask] * attention_weights[mask], dim=0)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")
        
        # 为每个节点获取其对应图的全局嵌入
        expanded_global_emb = global_emb[batch]
        
        # 结合局部节点嵌入和全局图嵌入
        combined_emb = torch.cat([node_emb, expanded_global_emb], dim=1)
        
        # 通过MLP生成最终嵌入
        final_emb = self.combiner(combined_emb)
        
        # 标准化嵌入
        final_emb = F.normalize(final_emb, p=2, dim=1)
        
        return final_emb
