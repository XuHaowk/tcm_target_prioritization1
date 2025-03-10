"""
增强的训练器

这个模块提供了用于训练图神经网络的增强训练器，
包括改进的训练循环、学习率调度和早停机制。
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

from .improved_loss import ImprovedContrastiveLoss

class EnhancedTrainer:
    """增强的训练器，用于训练图神经网络模型"""
    
    def __init__(self, model, device, num_epochs=100, batch_size=64, learning_rate=0.001,
                 weight_decay=1e-5, margin=0.5, neg_samples=2, save_model=True, 
                 model_path='results/models/best_model.pt', patience=10):
        """
        初始化训练器
        
        参数:
            model: 图神经网络模型
            device: 设备 (CPU/GPU)
            num_epochs: 训练轮数
            batch_size: 批处理大小
            learning_rate: 学习率
            weight_decay: 权重衰减
            margin: 对比损失的边界
            neg_samples: 每个正样本的负样本数量
            save_model: 是否保存模型
            model_path: 模型保存路径
            patience: 早停耐心值
        """
        self.model = model
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.margin = margin
        self.neg_samples = neg_samples
        self.save_model = save_model
        self.model_path = model_path
        self.patience = patience
        
        # 初始化损失函数
        self.loss_fn = ImprovedContrastiveLoss(margin=margin, temperature=0.1, adaptive_temp=True)
        
        print(f"Trainer initialized with device: {device}")
    
    def _generate_training_pairs(self, data, important_targets):
        """
        生成正负训练对
        
        参数:
            data: PyG数据对象
            important_targets: 重要靶点列表
            
        返回:
            元组 (正对，负对)
        """
        # 提取边索引
        edge_index = data.edge_index.cpu()
        
        # 提取节点类型
        compound_indices = data.compound_indices.cpu()
        target_indices = data.target_indices.cpu()
        
        # 创建现有边集合
        existing_edges = set()
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            existing_edges.add((src, dst))
            existing_edges.add((dst, src))  # 对于无向图
        
        # 生成正对
        pos_pairs = []
        
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            
            # 只包括化合物-靶点对
            if (src in compound_indices and dst in target_indices) or \
               (src in target_indices and dst in compound_indices):
                # 添加标签1（正例）
                pos_pairs.append([src, dst, 1])
        
        # 生成负对
        neg_pairs = []
        
        # 负样本数量
        num_neg_samples = len(pos_pairs) * self.neg_samples
        
        # 采样负对
        while len(neg_pairs) < num_neg_samples:
            # 随机采样化合物和靶点
            src_idx = np.random.choice(len(compound_indices))
            dst_idx = np.random.choice(len(target_indices))
            
            src = compound_indices[src_idx].item()
            dst = target_indices[dst_idx].item()
            
            # 跳过已存在的边
            if (src, dst) in existing_edges:
                continue
            
            # 添加标签0（负例）
            neg_pairs.append([src, dst, 0])
        
        # 转换为张量
        pos_pairs = torch.tensor(pos_pairs, dtype=torch.long)
        neg_pairs = torch.tensor(neg_pairs, dtype=torch.long)
        
        print(f"Generated {len(pos_pairs)} positive pairs and {len(neg_pairs)} negative pairs")
        
        return pos_pairs, neg_pairs
    
    def _create_dataloader(self, pos_pairs, neg_pairs):
        """
        创建数据加载器
        
        参数:
            pos_pairs: 正对
            neg_pairs: 负对
            
        返回:
            DataLoader
        """
        # 合并正负对
        all_pairs = torch.cat([pos_pairs, neg_pairs], dim=0)
        
        # 随机打乱
        idx = torch.randperm(all_pairs.shape[0])
        all_pairs = all_pairs[idx]
        
        # 创建数据集
        dataset = TensorDataset(all_pairs)
        
        # 创建数据加载器
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        return dataloader
    
    def train(self, data, important_targets):
        """
        Train the model
    
        Parameters:
            data: PyG data object
            important_targets: List of important target indices
        
        Returns:
            Node embeddings
        """
        self.model.train()
    
        # Move data to device
        data = data.to(self.device)
    
        # Generate training pairs
        pos_pairs, neg_pairs = self._generate_training_pairs(data, important_targets)
        print(f"Number of positive pairs: {len(pos_pairs)}")
    
        # Create data loader
        train_loader = self._create_dataloader(pos_pairs, neg_pairs)
    
        # Create optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs,
                                                              eta_min=self.learning_rate * 0.01)
    
        # Early stopping setup
        best_loss = float('inf')
        patience_counter = 0
    
        # Training loop
        for epoch in range(self.num_epochs):
            total_loss = 0.0
        
            # Train on each batch
            for batch in train_loader:
                batch = batch[0].to(self.device)
            
                # Zero gradients
                optimizer.zero_grad()
            
                # Forward pass - check model type to determine what arguments to pass
                if hasattr(data, 'batch') and hasattr(self.model, 'pooling'):
                    # This is likely a GraphPoolRGCN model that needs batch information
                    embeddings = self.model(data.x, data.edge_index, data.edge_type, data.batch, data.edge_weight)
                else:
                    # Standard model without batch parameter
                    embeddings = self.model(data.x, data.edge_index, data.edge_type, data.edge_weight)
            
                # Calculate loss
                loss = self.loss_fn(embeddings, batch, hard_negative_mining=(epoch > self.num_epochs // 3))
            
                # Backward pass
                loss.backward()
            
                # Update weights
                optimizer.step()
            
                total_loss += loss.item()
        
            # Update learning rate
            scheduler.step()
        
            # Print training progress
            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            
                # Save best model
                if self.save_model:
                    os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                    torch.save(self.model.state_dict(), self.model_path)
                    print(f"Model saved to {self.model_path}")
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
        # Load best model
        if self.save_model and os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
    
        # Generate final embeddings
        self.model.eval()
        with torch.no_grad():
            # Again, check model type to determine what arguments to pass
            if hasattr(data, 'batch') and hasattr(self.model, 'pooling'):
                embeddings = self.model(data.x, data.edge_index, data.edge_type, data.batch, data.edge_weight)
            else:
                embeddings = self.model(data.x, data.edge_index, data.edge_type, data.edge_weight)
    
        return embeddings
