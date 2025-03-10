"""
注意力增强的关系图卷积网络（AttentionRGCN）模型

这个模块提供带有图注意力机制的RGCN实现，能够更好地
捕获不同关系类型和邻居节点的重要性。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, GATConv

class AttentionRGCN(nn.Module):
    """
    注意力增强的RGCN模型，结合了关系感知和图注意力机制
    """
    def __init__(self, in_dim, hidden_dims, out_dim, num_relations, num_bases=None, 
                 dropout=0.2, residual=True, layer_norm=True):
        """
        初始化AttentionRGCN模型

        参数:
            in_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表 (可变深度)
            out_dim: 输出嵌入维度
            num_relations: 关系类型数量
            num_bases: 基矩阵数量
            dropout: Dropout概率
            residual: 是否使用残差连接
            layer_norm: 是否使用层归一化
        """
        super(AttentionRGCN, self).__init__()
        
        # 初始化参数
        self.residual = residual
        self.layer_norm = layer_norm
        
        # 确保hidden_dims是列表
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        
        # 构建多层网络
        self.layers = nn.ModuleList()
        
        # 输入层 (RGCN)
        if num_bases is None:
            num_bases = min(num_relations, hidden_dims[0])
        
        self.layers.append(RGCNConv(in_dim, hidden_dims[0], num_relations, num_bases))
        
        # 添加层归一化
        self.layer_norms = nn.ModuleList()
        if layer_norm:
            self.layer_norms.append(nn.LayerNorm(hidden_dims[0]))
        
        # 构建隐藏层 (RGCN和GAT混合)
        for i in range(len(hidden_dims) - 1):
            # RGCN层处理关系
            self.layers.append(RGCNConv(hidden_dims[i], hidden_dims[i+1], 
                                        num_relations, num_bases))
            
            # GAT层添加注意力
            self.layers.append(GATConv(hidden_dims[i+1], hidden_dims[i+1], 
                                       heads=4, concat=False, dropout=dropout))
            
            # 添加层归一化
            if layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_dims[i+1]))
                self.layer_norms.append(nn.LayerNorm(hidden_dims[i+1]))
        
        # 输出层
        self.layers.append(RGCNConv(hidden_dims[-1], out_dim, num_relations, num_bases))
        
        # Dropout层
        self.dropout = dropout
        
        # 初始化权重
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化网络权重"""
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def forward(self, x, edge_index, edge_type, edge_weight=None):
        """
        前向传播

        参数:
            x: 节点特征
            edge_index: 边索引
            edge_type: 边类型
            edge_weight: 边权重
            
        返回:
            节点嵌入
        """
        prev_x = None  # 用于残差连接
        norm_idx = 0   # 层归一化索引
        
        # 通过各层传播
        for i, layer in enumerate(self.layers):
            # 保存前一层输出用于残差连接
            if i > 0 and self.residual and prev_x is not None and prev_x.shape == x.shape:
                prev_x = x
            
            # RGCN层
            if isinstance(layer, RGCNConv):
                x = layer(x, edge_index, edge_type)
                
                # 添加层归一化
                if self.layer_norm and norm_idx < len(self.layer_norms):
                    x = self.layer_norms[norm_idx](x)
                    norm_idx += 1
                
                # 添加残差连接
                if i > 0 and self.residual and prev_x is not None and prev_x.shape == x.shape:
                    x = x + prev_x
                
                # 添加非线性和Dropout (对于中间层)
                if i < len(self.layers) - 1:
                    x = F.leaky_relu(x, negative_slope=0.1)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            
            # GAT层
            elif isinstance(layer, GATConv):
                x = layer(x, edge_index)
                
                # 添加层归一化
                if self.layer_norm and norm_idx < len(self.layer_norms):
                    x = self.layer_norms[norm_idx](x)
                    norm_idx += 1
                
                # 添加非线性和Dropout
                x = F.leaky_relu(x, negative_slope=0.1)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 标准化最终嵌入
        x = F.normalize(x, p=2, dim=1)
        
        return x
