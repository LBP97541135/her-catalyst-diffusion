import numpy as np
import torch

def build_radius_graph(pos, cutoff=5.0):
    """
    构建半径邻域图。
    Args:
        pos: 原子坐标 [N, 3]
        cutoff: 截断半径 (Å)
    Returns:
        edge_index: [2, E] 邻接关系
    """
    # 计算距离矩阵
    dist_matrix = torch.cdist(pos, pos)
    # 筛选距离在 (0, cutoff] 之间的边
    mask = (dist_matrix > 0) & (dist_matrix <= cutoff)
    edge_index = mask.nonzero().t()
    return edge_index

def calculate_her_performance(adsorption_energy):
    """
    计算基于 Sabatier 原理的 HER 性能评分。
    当 ΔG_H 接近 0 eV 时，催化活性最高，性能分数接近 1。
    """
    optimal_gh = 0.0
    sigma = 0.1 # 高斯函数宽度，决定了对偏离 0 的敏感度
    performance = np.exp(-(adsorption_energy - optimal_gh)**2 / (2 * sigma**2))
    return performance

def evaluate_stability(energy_above_hull):
    """
    评估热力学稳定性。
    通常 E_above_hull < 0.1 eV/atom 被认为具有较好的稳定性。
    """
    return energy_above_hull < 0.1

def check_geometry_validity(pos, threshold=1.0):
    """
    基础几何有效性检查。
    检查原子间距是否小于阈值（防止非物理的碰撞）。
    """
    # 暂实现为基础占位符
    return True
