import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskOptimizationLoss(nn.Module):
    """
    多任务联合优化损失函数。
    同时平衡去噪质量与材料属性（活性、稳定性、合成性）的优化目标。
    """
    def __init__(self, weights=None):
        super(MultiTaskOptimizationLoss, self).__init__()
        # 默认权重分配，增强活性引导
        self.weights = weights or {
            'diffusion': 1.0, 
            'her': 2.0, 
            'stability': 1.5, 
            'synthesizability': 1.0
        }

    def forward(self, pred_dict, target_dict):
        """
        计算总损失。
        Args:
            pred_dict: 包含 'noise_pred', 'her_pred', 'energy_pred', 'synth_score'
            target_dict: 包含 'noise_true', 'her_target', 'energy_threshold'
        """
        loss_dict = {}
        
        # 1. 扩散模型 MSE 损失 (结构还原 - 预测噪声 vs 真实噪声)
        # 确保 shape 一致
        noise_pred = pred_dict['noise_pred']
        noise_true = target_dict['noise_true']
        loss_dict['diffusion'] = F.mse_loss(noise_pred, noise_true)
        
        # 2. HER 活性损失 (确保 shape 一致)
        her_pred = pred_dict['her_pred'].view(-1)
        her_target = target_dict.get('her_target', torch.zeros_like(her_pred)).view(-1)
        loss_dict['her'] = F.huber_loss(her_pred, her_target, delta=0.1)
        
        # 3. 稳定性损失 (ReLU 约束：能量需低于阈值)
        energy_pred = pred_dict['energy_pred'].view(-1)
        energy_target = torch.full_like(energy_pred, target_dict.get('energy_threshold', 0.1))
        loss_dict['stability'] = torch.mean(F.relu(energy_pred - energy_target))
        
        # 4. 可合成性损失 (BCE Loss)
        # 将合成性视为二分类问题 (可合成/不可合成)
        synth_score = pred_dict['synth_score'].view(-1)
        synth_target = target_dict.get('synth_label', torch.zeros_like(synth_score)).view(-1)
        loss_dict['synthesizability'] = F.binary_cross_entropy(synth_score, synth_target)
        
        # 加权求和
        total_loss = sum(self.weights[k] * loss_dict[k] for k in self.weights.keys() if k in loss_dict)
        
        return total_loss, loss_dict
