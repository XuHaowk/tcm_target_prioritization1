"""
基础关系图卷积网络（RGCN）模型

这个模块提供了RGCN的基础实现，用于处理带有边类型的异质图。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class RGCN(nn.Module):
    """
    关系图卷积网络（RGCN）模型
    处理带有不同类型关系的异质图
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations, num_bases=None, dropout=0.2):
        """
        初始化RGCN模型

        参数:
            in_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            out_dim: 输出嵌入维度
            num_relations: 关系类型数量
            num_bases: 基矩阵数量（用于减少参数）
            dropout: Dropout概率
        """
        super(RGCN, self).__init__()
        
        # 设置基矩阵数量
        if num_bases is None:
            num_bases = min(num_relations, hidden_dim)
        
        # 定义RGCN层
        self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases)
        self.conv3 = RGCNConv(hidden_dim, out_dim, num_relations=num_relations, num_bases=num_bases)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout概率
        self.dropout = dropout
        
        # 初始化权重
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=0.1)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x, edge_index, edge_type, edge_weight=None):
        """
        前向传播

        参数:
            x: 节点特征
            edge_index: 边索引
            edge_type: 边类型
            edge_weight: 边权重（RGCN不使用，但保留兼容性）

        返回:
            节点嵌入
        """
        # 第一层
        x = self.conv1(x, edge_index, edge_type)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层
        x = self.conv2(x, edge_index, edge_type)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第三层
        x = self.conv3(x, edge_index, edge_type)
        
        # 归一化输出嵌入
        x = F.normalize(x, p=2, dim=1)
        
        return x
