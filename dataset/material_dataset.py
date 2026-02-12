import torch
from torch.utils.data import Dataset
import json
import os
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from config import DATA_CONFIG

class MaterialDataset(Dataset):
    """
    材料数据集加载器。
    对接原项目的 CIF 晶体结构文件及其 2D 筛选评分结果。
    支持合并多个来源的数据（用于主动学习）。
    """
    def __init__(self, data_sources=None):
        """
        data_sources: List of tuples (raw_cif_dir, results_json_path)
        """
        if data_sources is None:
            data_sources = [(DATA_CONFIG["raw_cif_dir"], DATA_CONFIG["label_json_path"])]
            
        self.valid_samples = []
        
        for raw_cif_dir, results_json_path in data_sources:
            if not os.path.exists(results_json_path):
                print(f"警告: 未找到标签文件: {results_json_path}，跳过该来源。")
                continue
                
            with open(results_json_path, 'r') as f:
                labels_data = json.load(f)
                
            label_map = {item['filename']: item for item in labels_data}
            for filename in os.listdir(raw_cif_dir):
                if filename.endswith('.cif') and filename in label_map:
                    self.valid_samples.append({
                        'cif_path': os.path.join(raw_cif_dir, filename),
                        'label_info': label_map[filename]
                    })
        
        print(f"数据集加载完成: 总计 {len(self.valid_samples)} 个有效样本。")

    def add_samples(self, raw_cif_dir, results_json_path, min_score=0.1):
        """
        动态添加新样本（主动学习反馈环）。
        """
        if not os.path.exists(results_json_path):
            return
            
        with open(results_json_path, 'r') as f:
            labels_data = json.load(f)
            
        label_map = {item['filename']: item for item in labels_data}
        added_count = 0
        
        for filename in os.listdir(raw_cif_dir):
            if filename.endswith('.cif') and filename in label_map:
                info = label_map[filename]
                # 质量筛选：只保留评分高于阈值的样本，或者是具有新颖性的样本
                if info.get('score', 0) >= min_score:
                    self.valid_samples.append({
                        'cif_path': os.path.join(raw_cif_dir, filename),
                        'label_info': info
                    })
                    added_count += 1
        
        print(f"主动学习反馈: 新增 {added_count} 个高质量真实样本。当前数据集总量: {len(self.valid_samples)}")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        """
        读取单个晶体样本并转换为张量格式。
        """
        sample = self.valid_samples[idx]
        cif_path = sample['cif_path']
        label_info = sample['label_info']
        
        # 使用 pymatgen 解析 CIF
        try:
            structure = Structure.from_file(cif_path)
        except Exception as e:
            # 如果解析失败，返回一个默认值或跳过（这里简单处理）
            print(f"解析 CIF 失败: {cif_path}, error: {e}")
            # 返回数据集中的第一个样本作为替代 (简单鲁棒性处理)
            return self.__getitem__(0) if idx != 0 else None
            
        # 原子序数 (z) 和 笛卡尔坐标 (pos)
        z = torch.tensor([site.specie.number for site in structure], dtype=torch.long)
        pos = torch.tensor(structure.cart_coords, dtype=torch.float)
        
        # 提取原项目评分作为模型引导标签
        her_label = torch.tensor([label_info.get('score', 0.5)], dtype=torch.float)
        energy_label = torch.tensor([label_info.get('binding_energy_estimate', 0.1)], dtype=torch.float)
        
        # 构造可合成性标签 (Synthesizability Label)
        # 逻辑: 能量越低且 HER 活性越好的材料，越有可能被实验关注和合成
        # 这里使用简单的启发式规则生成伪标签，用于演示多任务学习
        synth_pseudo_label = 1.0 if (label_info.get('binding_energy_estimate', 0.1) < 0.15) else 0.0
        synth_label = torch.tensor([synth_pseudo_label], dtype=torch.float)
        
        return {
            'z': z,
            'pos': pos,
            'her_label': her_label,
            'energy_label': energy_label,
            'synth_label': synth_label,
            'formula': structure.composition.reduced_formula
        }

def collate_fn(batch):
    """
    处理变长图数据的批处理函数
    """
    return {
        'z': [item['z'] for item in batch],
        'pos': [item['pos'] for item in batch],
        'her_label': torch.stack([item['her_label'] for item in batch]),
        'energy_label': torch.stack([item['energy_label'] for item in batch]),
        'synth_label': torch.stack([item['synth_label'] for item in batch]),
        'formulas': [item['formula'] for item in batch]
    }
