import torch
import os
import numpy as np
from models.diffusion_model import DiffusionModel
from utils.geo_utils import calculate_her_performance, build_radius_graph
from utils.vis import plot_her_distribution, plot_stability_curve, plot_structure_grid, plot_her_vs_stability
from pymatgen.core import Structure, Lattice
from config import DATA_CONFIG, MODEL_CONFIG, DIFFUSION_CONFIG

from models.structure_generator import StructureGenerator

def test():
    """
    测试脚本入口：生成新材料并生成可视化报告。
    """
    print("=== 启动测试与可视化生成流水线 ===")
    
    # 1. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffusionModel(num_layers=MODEL_CONFIG["num_layers"], 
                           feat_dim=MODEL_CONFIG["feat_dim"]).to(device)
    
    checkpoint_path = os.path.join(DATA_CONFIG["checkpoints_dir"], 'diffusion_guided_v1.pth')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"成功加载模型权重: {checkpoint_path}")
    else:
        print("警告: 未找到模型权重，将使用随机初始化模型生成。")
        
    # 2. 采样生成 (使用 StructureGenerator)
    print("初始化结构生成器...")
    generator = StructureGenerator(model, device)
    
    print("开始执行反向扩散采样...")
    cif_paths = generator.generate(num_samples=15, save_dir='results')
    
    # 3. 属性预测
    print("预测生成结构属性...")
    results = generator.predict_properties(cif_paths)
    
    gh_values = [r['her_pred'] for r in results]
    stability_scores = [r['energy_pred'] for r in results]
    
    # 3. 运行可视化优化
    print("生成可视化图表...")
    os.makedirs('results', exist_ok=True)
    
    # 图 1: HER 活性分布
    plot_her_distribution(gh_values, 'results/her_performance.png')
    
    # 图 2: 稳定性分布
    plot_stability_curve(stability_scores, 'results/stability_curve.png')
    
    # 图 3: 联合性能分布
    plot_her_vs_stability(gh_values, stability_scores, 'results/property_joint_dist.png')
    
    # 图 4: 生成结构预览
    if cif_paths:
        plot_structure_grid(cif_paths, 'results/generated_structures.png')
    
    print(f"可视化报告已生成至 results/ 目录。")
    print(f"生成样本总数: {len(cif_paths)}")
    print(f"平均 ΔG_H: {np.mean(gh_values):.4f} eV")
    if gh_values:
        print(f"最佳 ΔG_H: {min(gh_values, key=lambda x: abs(x)):.4f} eV")

if __name__ == "__main__":
    test()
