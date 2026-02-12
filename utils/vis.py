import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ase.visualize.plot import plot_atoms
from ase.io import read
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

# 设置全局绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

def plot_her_distribution(gh_values, save_path):
    """
    绘制 ΔG_H 分布直方图。
    红色虚线代表理想活性中心 (0 eV)。
    """
    plt.figure(figsize=(10, 6))
    
    # 使用核密度估计和直方图组合
    sns.histplot(gh_values, kde=True, stat="density", color='royalblue', 
                 edgecolor='white', alpha=0.6, label='Generated Materials')
    sns.kdeplot(gh_values, color='darkblue', linewidth=2)
    
    plt.axvline(0, color='crimson', linestyle='--', linewidth=2, label='Ideal Active Site (0 eV)')
    plt.axvspan(-0.1, 0.1, color='green', alpha=0.1, label='Optimal Range (±0.1 eV)')
    
    plt.title('HER Catalytic Activity Distribution ($\Delta G_H$)', fontsize=16, fontweight='bold')
    plt.xlabel('Gibbs Free Energy $\Delta G_H$ (eV)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(frameon=True, fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_stability_curve(energies, save_path):
    """
    绘制生成样本的稳定性评估散点图。
    """
    plt.figure(figsize=(10, 6))
    
    x = range(1, len(energies) + 1)
    
    # 区分稳定和不稳定的点
    stable_mask = np.array(energies) <= 0.1
    unstable_mask = ~stable_mask
    
    plt.scatter(np.array(x)[stable_mask], np.array(energies)[stable_mask], 
                color='forestgreen', s=100, label='Stable (<= 0.1 eV/atom)', edgecolors='white', linewidth=1.5)
    plt.scatter(np.array(x)[unstable_mask], np.array(energies)[unstable_mask], 
                color='firebrick', s=100, label='Unstable (> 0.1 eV/atom)', edgecolors='white', linewidth=1.5)
    
    plt.axhline(0.1, color='gray', linestyle='--', linewidth=2, label='Stability Threshold')
    
    plt.title('Thermodynamic Stability Assessment ($E_{hull}$)', fontsize=16, fontweight='bold')
    plt.xlabel('Generated Sample Index', fontsize=14)
    plt.ylabel('Energy Above Hull (eV/atom)', fontsize=14)
    plt.legend(frameon=True, fontsize=12)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_loss_curve(loss_history, save_path):
    """
    绘制模型训练的损失下降曲线。
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制原始损失
    plt.plot(loss_history, color='lightgray', linewidth=1, alpha=0.5, label='Raw Loss')
    
    # 绘制平滑曲线
    if len(loss_history) > 10:
        window_size = max(5, len(loss_history) // 20)
        smooth_loss = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
        x_smooth = range(window_size-1, len(loss_history))
        plt.plot(x_smooth, smooth_loss, color='royalblue', linewidth=2.5, label=f'Smoothed (MA-{window_size})')
        
    plt.title('Training Convergence: Multi-Task Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Training Epochs', fontsize=14)
    plt.ylabel('Total Loss (Log Scale)', fontsize=14)
    plt.yscale('log')
    plt.legend(frameon=True, fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_multi_task_comparison(metrics_dict, save_path):
    """
    绘制多任务性能对比图 (例如不同迭代下的精度提升)。
    metrics_dict: {'Iteration 1': 0.15, 'Iteration 2': 0.08, ...}
    """
    plt.figure(figsize=(10, 6))
    iters = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    colors = sns.color_palette("viridis", len(iters))
    plt.bar(iters, values, color=colors, alpha=0.8)
    
    # 在柱状图上标注数值
    for i, v in enumerate(values):
        plt.text(i, v + 0.005, f"{v:.4f}", ha='center', fontsize=11, fontweight='bold')
        
    plt.title('Active Learning Performance Evolution (MAE)', fontsize=14)
    plt.ylabel('Mean Absolute Error (eV)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_her_vs_stability(gh_values, stability_values, save_path):
    """
    绘制 HER 活性与稳定性的联合分布散点图。
    """
    plt.figure(figsize=(10, 8))
    # 创建密度背景
    sns.kdeplot(x=gh_values, y=stability_values, levels=5, color="gray", alpha=0.3)
    # 散点图
    scatter = plt.scatter(gh_values, stability_values, c=gh_values, cmap='coolwarm', 
                         edgecolor='white', s=80, alpha=0.8)
    
    plt.axvline(0, color='red', linestyle='--', alpha=0.6, label='Ideal HER (0 eV)')
    plt.axhline(0.1, color='green', linestyle=':', alpha=0.6, label='Stability Threshold')
    
    plt.colorbar(scatter, label='ΔG_H (eV)')
    plt.title('Property Joint Distribution: HER Activity vs. Stability', fontsize=14)
    plt.xlabel('ΔG_H (eV)', fontsize=12)
    plt.ylabel('Energy Above Hull (eV/atom)', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_structure_grid(cif_paths, save_path):
    """
    将生成的多个材料结构可视化并排列成网格。
    """
    num_structures = len(cif_paths)
    if num_structures == 0:
        return

    cols = 4
    rows = (num_structures + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.5, rows*3.5))
    if num_structures == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, cif_path in enumerate(cif_paths):
        try:
            struct = Structure.from_file(cif_path)
            atoms = AseAtomsAdaptor.get_atoms(struct)
            
            # 使用 ASE 的 plot_atoms
            # 调整视角以展示层状结构
            plot_atoms(atoms, axes[i], radii=0.4, rotation=('10x,10y,0z'))
            
            # 提取化学式作为标题
            formula = struct.composition.reduced_formula
            axes[i].set_title(f'{formula}\n(Sample {i+1})', fontsize=11, fontweight='bold')
            axes[i].axis('off')
            
            # 添加边框
            for spine in axes[i].spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('#dddddd')
        except Exception as e:
            print(f"Error plotting {cif_path}: {e}")
            axes[i].text(0.5, 0.5, "Error", ha='center', va='center')
            axes[i].axis('off')
        
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle('Generated 2D Material Candidates', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
