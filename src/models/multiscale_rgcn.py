"""
多尺度关系图卷积网络（MultiscaleRGCN）模型

这个模块提供了多尺度RGCN实现，能够通过注意力机制
整合不同层次的图结构信息。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class MultiscaleRGCN(nn.Module):
    """
    多尺度RGCN模型，通过跳跃连接整合不同层次的图结构信息
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations, num_layers=3, 
                 num_bases=None, dropout=0.2):
        """
        初始化MultiscaleRGCN模型

        参数:
            in_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            out_dim: 输出嵌入维度
            num_relations: 关系类型数量
            num_layers: RGCN层数
            num_bases: 基矩阵数量
            dropout: Dropout概率
        """
        super(MultiscaleRGCN, self).__init__()
        
        # 初始化参数
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        
        # 输入投影
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # RGCN层
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases))
        
        # 批归一化层
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # 用于结合多尺度特征的注意力
        self.scale_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        
        # Dropout
        self.dropout = dropout
        
        # 初始化权重
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化网络权重"""
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            self.input_proj.bias.data.zero_()
        
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            self.output_proj.bias.data.zero_()
        
        for m in self.scale_attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        
        for conv in self.convs:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()
    
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
        # 输入投影
        x = self.input_proj(x)
        
        # 存储不同尺度的表示
        scale_representations = []
        
        # 逐层处理
        for i in range(self.num_layers):
            # 图卷积
            x = self.convs[i](x, edge_index, edge_type)
            x = self.bns[i](x)
            x = F.leaky_relu(x, negative_slope=0.1)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # 保存此尺度的表示
            scale_representations.append(x)
        
        # 通过注意力机制整合不同尺度的表示
        attention_scores = []
        for rep in scale_representations:
            score = self.scale_attention(rep)
            attention_scores.append(score)
        
        # 将注意力分数拼接起来处理
        attention_scores = torch.cat(attention_scores, dim=1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # 加权组合不同尺度的表示
        multi_scale_emb = torch.zeros_like(scale_representations[0])
        for i, rep in enumerate(scale_representations):
            multi_scale_emb += rep * attention_weights[:, i].unsqueeze(1)
        
        # 输出投影
        out = self.output_proj(multi_scale_emb)
        
        # 标准化嵌入
        out = F.normalize(out, p=2, dim=1)
        
        return out
