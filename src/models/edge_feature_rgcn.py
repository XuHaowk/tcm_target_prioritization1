"""
边特征增强的关系图卷积网络（EdgeFeatureRGCN）模型

这个模块提供了能够处理边特征和权重的RGCN实现，
能够更全面地利用边的语义信息。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class EdgeFeatureRGCN(nn.Module):
    """
    边特征感知的RGCN模型，能够利用边的权重和特征
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations, edge_dim=None, 
                 num_bases=None, dropout=0.2):
        """
        初始化EdgeFeatureRGCN模型

        参数:
            in_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            out_dim: 输出嵌入维度
            num_relations: 关系类型数量
            edge_dim: 边特征维度 (若无则为None)
            num_bases: RGCN基矩阵数量
            dropout: Dropout概率
        """
        super(EdgeFeatureRGCN, self).__init__()
        
        # 关系嵌入表示
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
        self.edge_encoder = None
        
        # 如果有边特征，添加边特征编码器
        if edge_dim is not None:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # RGCN层
        self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases)
        self.conv3 = RGCNConv(hidden_dim, out_dim, num_relations=num_relations, num_bases=num_bases)
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout
        self.dropout = dropout
        
        # 初始化权重
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化网络权重"""
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        if self.edge_encoder is not None:
            for m in self.edge_encoder.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()
        
        for m in [self.conv1, self.conv2, self.conv3]:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
    
    def forward(self, x, edge_index, edge_type, edge_attr=None, edge_weight=None):
        """
        前向传播

        参数:
            x: 节点特征
            edge_index: 边索引
            edge_type: 边类型
            edge_attr: 边特征 (可选)
            edge_weight: 边权重 (可选)
            
        返回:
            节点嵌入
        """
        # 获取关系嵌入
        rel_emb = self.relation_embedding(edge_type)
        
        # 如果有边特征，将其与关系嵌入结合
        if edge_attr is not None and self.edge_encoder is not None:
            edge_emb = self.edge_encoder(edge_attr)
            rel_emb = rel_emb + edge_emb
        
        # 如果有边权重，将其应用到关系嵌入
        if edge_weight is not None:
            rel_emb = rel_emb * edge_weight.view(-1, 1)
        
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
        
        # 标准化嵌入
        x = F.normalize(x, p=2, dim=1)
        
        return x
