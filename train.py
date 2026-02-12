import torch
from torch.utils.data import DataLoader
from models.diffusion_model import DiffusionModel
from models.optimization import MultiTaskOptimizationLoss
from dataset.material_dataset import MaterialDataset, collate_fn
from utils.vis import plot_loss_curve
from utils.diffusion_utils import GaussianDiffusion
from utils.geo_utils import build_radius_graph
from config import DATA_CONFIG, TRAIN_CONFIG, MODEL_CONFIG, DIFFUSION_CONFIG
import os
import sys
import numpy as np
from pathlib import Path

# 添加原项目路径以便调用其评估逻辑
ORIGINAL_PROJECT_PATH = DATA_CONFIG["original_project_path"]
sys.path.insert(0, str(Path(ORIGINAL_PROJECT_PATH) / "src"))

from data.filter_2d_materials_secondary import TwoDMaterialScreener

from models.structure_generator import StructureGenerator

def run_active_learning(model, device, num_samples=20):
    """
    主动学习：使用当前模型生成样本，并利用原项目评估逻辑获取“真实”标签。
    """
    print(f"\n--- 启动主动学习循环: 生成并评估 {num_samples} 个新样本 ---")
    model.eval()
    temp_dir = os.path.join(DATA_CONFIG["results_dir"], "active_learning_temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 1. 生成样本
    generator = StructureGenerator(model, device)
    generator.generate(num_samples=num_samples, save_dir=temp_dir)
    
    # 2. 调用原项目评估逻辑
    eval_output_dir = os.path.join(DATA_CONFIG["results_dir"], "active_learning_eval")
    os.makedirs(eval_output_dir, exist_ok=True)
    screener = TwoDMaterialScreener(temp_dir, eval_output_dir)
    
    try:
        screener.run_screening()
        results_json = os.path.join(eval_output_dir, "2d_screening_results.json")
        if os.path.exists(results_json):
            import json
            with open(results_json, 'r') as f:
                new_labels = json.load(f)
            print(f"主动学习完成: 成功获取 {len(new_labels)} 个新真实样本标签。")
            return temp_dir, results_json
    except Exception as e:
        print(f"主动学习评估失败: {e}")
    
    return None, None

def train():
    """
    模型训练主循环。
    通过真实样本驱动和主动学习进一步提升效果。
    """
    print("=== 开始训练真实样本驱动的扩散模型 (Active Learning Enhanced) ===")
    
    # 1. 配置初始真实样本路径
    data_sources = [
        (DATA_CONFIG["raw_cif_dir"], DATA_CONFIG["label_json_path"])
    ]
    
    # 2. 初始数据加载
    dataset = MaterialDataset(data_sources)

    # 3. 模型与扩散器初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffusionModel(num_layers=MODEL_CONFIG["num_layers"], 
                           feat_dim=MODEL_CONFIG["feat_dim"]).to(device)
    
    diffusion = GaussianDiffusion(
        num_steps=DIFFUSION_CONFIG["num_steps"],
        beta_start=DIFFUSION_CONFIG["beta_start"],
        beta_end=DIFFUSION_CONFIG["beta_end"],
        schedule_type=DIFFUSION_CONFIG["schedule_type"]
    ).get_params(device)

    checkpoint_path = os.path.join(DATA_CONFIG["checkpoints_dir"], 'diffusion_guided_v1.pth')
    if os.path.exists(checkpoint_path):
        print(f"尝试加载预训练权重...")
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        except:
            print("架构不匹配，将从头开始训练。")
            
    criterion = MultiTaskOptimizationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG["lr"])

    # 4. 训练与主动学习循环
    num_iterations = TRAIN_CONFIG["num_iterations"]
    epochs_per_iter = TRAIN_CONFIG["epochs_per_iter"]
    all_loss_history = []
    
    for iter_idx in range(num_iterations):
        print(f"\n>> 主动学习迭代 {iter_idx+1}/{num_iterations}")
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
        
        for epoch in range(epochs_per_iter):
            model.train()
            epoch_loss = 0
            for batch in dataloader:
                for i in range(len(batch['z'])):
                    optimizer.zero_grad()
                    z = batch['z'][i].to(device)
                    pos_0 = batch['pos'][i].to(device)
                    her_label = batch['her_label'][i].to(device)
                    
                    # 采样随机时间步
                    t = torch.randint(0, diffusion.num_steps, (1,), device=device).long()
                    
                    # 前向加噪
                    noise = torch.randn_like(pos_0)
                    pos_t = diffusion.q_sample(pos_0, t, noise=noise)
                    
                    # 构建半径邻域图
                    edge_index = build_radius_graph(pos_t, cutoff=MODEL_CONFIG["radius_cutoff"])
                    
                    # 获取模型预测
                    pred_dict = model(z, pos_t, edge_index, t)
                    
                    # 准备训练标签
                    synth_label = batch['synth_label'][i].to(device)
                    
                    # 多任务损失计算
                    loss, loss_dict = criterion(pred_dict, {
                        'noise_true': noise,
                        'her_target': her_label,
                        'energy_threshold': 0.1,
                        'synth_label': synth_label
                    })
                    
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataset)
            all_loss_history.append(avg_loss)
            if (epoch + 1) % 10 == 0:
                print(f"Iteration {iter_idx+1}, Epoch {epoch+1}/{epochs_per_iter}, Loss: {avg_loss:.6f}")

        # 5. 执行主动学习，获取新样本并更新数据集
        if iter_idx < num_iterations - 1:
            new_cif_dir, new_labels_json = run_active_learning(model, device, num_samples=10)
            if new_cif_dir and new_labels_json:
                print(f"合并新生成的真实样本到训练集...")
                data_sources.append((new_cif_dir, new_labels_json))
                dataset = MaterialDataset(data_sources) # 重新加载数据集

    # 6. 保存最终模型与绘制损失曲线
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"训练完成，模型已保存至 {checkpoint_path}")
    
    plot_loss_curve(all_loss_history, 'results/active_learning_loss.png')

if __name__ == "__main__":
    train()
