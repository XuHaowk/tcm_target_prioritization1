"""
改进的对比学习损失函数

这个模块提供了用于训练图神经网络的改进对比学习损失函数，
包括硬负样本挖掘和自适应温度机制。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImprovedContrastiveLoss(nn.Module):
    """
    改进的对比学习损失函数，结合硬负样本挖掘和自适应温度
    """
    def __init__(self, margin=0.5, temperature=0.1, adaptive_temp=True):
        """
        初始化改进的对比学习损失函数
        
        参数:
            margin: 对比损失的边界
            temperature: 温度参数
            adaptive_temp: 是否使用自适应温度
        """
        super(ImprovedContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        self.adaptive_temp = adaptive_temp
        
        # 用于自适应温度的参数
        if adaptive_temp:
            self.log_temp = nn.Parameter(torch.tensor(np.log(temperature)))
    
    def forward(self, embeddings, pairs, hard_negative_mining=True):
        """
        计算对比损失
        
        参数:
            embeddings: 节点嵌入
            pairs: 标记对 (idx1, idx2, label)
            hard_negative_mining: 是否使用困难负样本挖掘
            
        返回:
            损失值
        """
        # 提取对和标签
        idx1 = pairs[:, 0]
        idx2 = pairs[:, 1]
        labels = pairs[:, 2].float()
        
        # 获取嵌入
        embed1 = embeddings[idx1]
        embed2 = embeddings[idx2]
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(embed1, embed2, dim=1)
        
        # 获取温度系数
        if self.adaptive_temp:
            temp = torch.exp(self.log_temp)
        else:
            temp = self.temperature
        
        # 正则化相似度
        sim_score = similarity / temp
        
        # 分离正样本和负样本
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        pos_pairs = pairs[pos_mask]
        neg_pairs = pairs[neg_mask]
        
        # 如果使用困难负样本挖掘
        if hard_negative_mining and torch.sum(neg_mask) > 0:
            # 计算所有负对的相似度
            neg_similarity = similarity[neg_mask]
            
            # 选择最困难的负样本（相似度最高的负样本）
            k = min(len(neg_similarity) // 2 + 1, len(neg_similarity))
            _, hard_indices = torch.topk(neg_similarity, k)
            hard_neg_mask = torch.zeros_like(neg_mask)
            hard_neg_mask[torch.nonzero(neg_mask).squeeze()[hard_indices]] = True
            
            # 组合正样本和困难负样本的掩码
            combined_mask = pos_mask | hard_neg_mask
            filtered_sim = similarity[combined_mask]
            filtered_labels = labels[combined_mask]
            
            # 计算InfoNCE损失
            pos_sim = filtered_sim[filtered_labels == 1] / temp
            neg_sim = filtered_sim[filtered_labels == 0] / temp
            
            # 正样本的对比损失
            if len(pos_sim) > 0:
                pos_loss = -torch.log(torch.exp(pos_sim) / 
                                    (torch.exp(pos_sim) + torch.sum(torch.exp(neg_sim))))
                pos_loss = pos_loss.mean()
            else:
                pos_loss = torch.tensor(0.0, device=embeddings.device)
            
            # 负样本的对比损失 (使用hinge损失)
            if len(neg_sim) > 0:
                neg_loss = torch.clamp(torch.exp(neg_sim) - torch.exp(torch.tensor(-self.margin) / temp), min=0).mean()
            else:
                neg_loss = torch.tensor(0.0, device=embeddings.device)
            
            # 总损失
            loss = pos_loss + 0.5 * neg_loss
        else:
            # 标准对比损失
            pos_loss = (1 - labels) * similarity.pow(2)
            neg_loss = labels * F.relu(self.margin - similarity).pow(2)
            
            loss = pos_loss + neg_loss
            loss = loss.mean()
        
        return loss
