import torch
import numpy as np
import os
from pymatgen.core import Structure, Lattice
from config import DATA_CONFIG, MODEL_CONFIG, DIFFUSION_CONFIG
from utils.geo_utils import build_radius_graph
from utils.diffusion_utils import GaussianDiffusion

class StructureGenerator:
    """
    基于扩散模型的二维材料结构生成器。
    封装了从高斯噪声到晶体结构的完整反向扩散过程。
    """
    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device else next(model.parameters()).device
        self.diffusion = GaussianDiffusion(
            num_steps=DIFFUSION_CONFIG["num_steps"],
            beta_start=DIFFUSION_CONFIG["beta_start"],
            beta_end=DIFFUSION_CONFIG["beta_end"],
            schedule_type=DIFFUSION_CONFIG["schedule_type"]
        ).get_params(self.device)
        
        # 常见二维材料元素 (Mo, S, C, N, B, W, Se, Ti, Nb)
        self.common_z = [42, 16, 6, 7, 5, 74, 34, 22, 41]

    def generate(self, num_samples=10, save_dir=None):
        """
        执行生成过程。
        Args:
            num_samples: 生成样本数量
            save_dir: CIF 文件保存路径 (可选)
        Returns:
            cif_paths: 生成的 CIF 文件路径列表
        """
        self.model.eval()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        cif_paths = []
        
        for i in range(num_samples):
            # 1. 初始化随机状态
            # 模拟生成 2D 材料典型的原子数 (例如 3-9 个原子，对应单胞)
            num_atoms = np.random.randint(3, 10)
            
            # 限制每个样本的元素种类不超过 3 种
            sample_elements = np.random.choice(self.common_z, size=np.random.randint(1, 4), replace=False)
            z = torch.tensor([np.random.choice(sample_elements) for _ in range(num_atoms)], dtype=torch.long, device=self.device)
            
            # 初始位置：高斯噪声
            # 特别初始化：在 xy 平面上稍微展开，z 轴方向压缩 (模拟 2D先验)
            pos = torch.randn((num_atoms, 3), device=self.device)
            pos[:, :2] *= 3.0
            pos[:, 2] *= 0.5
            
            # 2. 反向扩散过程
            # 使用标准的 p_sample 循环，而不是简单的梯度更新
            for t in reversed(range(self.diffusion.num_steps)):
                # 构建图
                edge_index = build_radius_graph(pos, cutoff=MODEL_CONFIG["radius_cutoff"])
                
                # 单步去噪
                t_tensor = torch.tensor([t], device=self.device).long()
                pos = self.diffusion.p_sample(self.model, pos, t_tensor, z, edge_index)
                
            # 3. 晶格重建与保存
            # 创建更符合 2D 材料特征的晶格 (较大的 a, b，适中的 c)
            lattice = Lattice.from_parameters(a=4.0, b=4.0, c=15.0, alpha=90, beta=90, gamma=120)
            # 注意：实际生成中应该同时学习晶格，这里简化为固定晶格或基于原子范围估算
            
            species = [int(zi) for zi in z.cpu()]
            struct = Structure(lattice, species, pos.cpu().numpy())
            
            if save_dir:
                cif_path = os.path.join(save_dir, f"generated_{i+1:04d}.cif")
                struct.to(fmt="cif", filename=cif_path)
                cif_paths.append(cif_path)
                
        return cif_paths

    def predict_properties(self, cif_paths):
        """
        对生成的结构进行属性预测 (HER, Stability, Synthesizability)。
        """
        results = []
        self.model.eval()
        
        for path in cif_paths:
            try:
                struct = Structure.from_file(path)
                z = torch.tensor([site.specie.number for site in struct], dtype=torch.long, device=self.device)
                pos = torch.tensor(struct.cart_coords, dtype=torch.float, device=self.device)
                edge_index = build_radius_graph(pos, cutoff=MODEL_CONFIG["radius_cutoff"])
                
                with torch.no_grad():
                    t = torch.zeros((1,), device=self.device).long()
                    pred = self.model(z, pos, edge_index, t)
                    
                results.append({
                    'cif_path': path,
                    'her_pred': pred['her_pred'].item(),
                    'energy_pred': pred['energy_pred'].item(),
                    'synth_score': pred['synth_score'].item()
                })
            except Exception as e:
                print(f"Error predicting properties for {path}: {e}")
                
        return results
