import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.absolute()

# 数据路径配置
DATA_CONFIG = {
    "original_project_path": "/Users/sealos/Desktop/程序/面试/material_generation",
    "raw_cif_dir": "/Users/sealos/Desktop/程序/面试/material_generation/data/raw",
    "label_json_path": "/Users/sealos/Desktop/程序/面试/material_generation/data/2d_materials/2d_screening_results.json",
    "results_dir": ROOT_DIR / "results",
    "checkpoints_dir": ROOT_DIR / "checkpoints",
}

# 扩散模型超参数
DIFFUSION_CONFIG = {
    "num_steps": 100,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "schedule_type": "cosine", # 切换为效果更平滑的余弦调度
}

# 模型架构配置
MODEL_CONFIG = {
    "num_layers": 6,
    "feat_dim": 128,
    "hidden_dim": 256,
    "radius_cutoff": 5.0, # Å
}

# 训练配置
TRAIN_CONFIG = {
    "lr": 1e-4,
    "batch_size": 32, # 提高 Batch Size 以稳定梯度
    "num_iterations": 5,
    "epochs_per_iter": 20,
    "device": "cuda", # 自动检测逻辑在代码中实现
}

# 确保目录存在
os.makedirs(DATA_CONFIG["results_dir"], exist_ok=True)
os.makedirs(DATA_CONFIG["checkpoints_dir"], exist_ok=True)
