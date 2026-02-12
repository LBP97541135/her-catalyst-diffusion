import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

def scatter_sum(src, index, dim=0, dim_size=None):
    """
    使用原生 PyTorch 实现 scatter_sum，确保跨平台兼容性。
    Args:
        src: 源张量
        index: 索引张量
        dim: 聚合维度
        dim_size: 目标张量的大小
    """
    if dim_size is None:
        dim_size = index.max().item() + 1
    out = torch.zeros(dim_size, *src.shape[1:], device=src.device, dtype=src.dtype)
    return out.index_add_(dim, index, src)

class EGNN_Layer(MessagePassing):
    """
    增强型 E(n) 等变图卷积层。
    加入了 LayerNorm 和残差连接以支持更深层的训练。
    支持注入时间步 Embedding。
    """
    def __init__(self, feat_dim, edge_dim=1, hidden_dim=256, time_emb_dim=None):
        super(EGNN_Layer, self).__init__(aggr='add')
        self.feat_dim = feat_dim
        
        # 时间步映射层
        if time_emb_dim:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, feat_dim)
            )
        else:
            self.time_mlp = None

        self.edge_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2 + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feat_dim)
        )
        
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        self.node_norm = nn.LayerNorm(feat_dim)
        
        nn.init.xavier_uniform_(self.coord_mlp[0].weight)
        nn.init.zeros_(self.coord_mlp[2].weight)

    def forward(self, h, pos, edge_index, edge_attr=None, t_emb=None):
        # 注入时间信息
        if self.time_mlp is not None and t_emb is not None:
            h = h + self.time_mlp(t_emb)

        row, col = edge_index
        rel_pos = pos[row] - pos[col]
        
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        dist_sq = dist**2
        rel_pos_norm = rel_pos / (dist + 1e-6)
        
        if edge_attr is None:
            edge_attr = dist_sq
        else:
            edge_attr = torch.cat([edge_attr, dist_sq], dim=-1)
            
        # 1. 生成消息 (包含相对位置信息)
        edge_feat = torch.cat([h[row], h[col], edge_attr], dim=-1)
        msg = self.edge_mlp(edge_feat)
        
        # 物理约束：基于距离的消息缩放 (避免近距离原子间的数值爆炸)
        msg_scale = torch.exp(-dist / 5.0) # 5.0 是 radius_cutoff
        msg = msg * msg_scale
        
        # 2. 更新坐标
        coord_weights = torch.tanh(self.coord_mlp(msg))
        trans = rel_pos_norm * coord_weights
        agg_trans = scatter_sum(trans, row, dim=0, dim_size=pos.size(0))
        pos = pos + agg_trans
        
        # 3. 更新节点特征 (带 LayerNorm 和残差)
        agg_msg = scatter_sum(msg, row, dim=0, dim_size=h.size(0))
        h_update = self.node_mlp(torch.cat([h, agg_msg], dim=-1))
        h = self.node_norm(h + h_update) # 残差 + 归一化
        
        return h, pos

class DiffusionModel(nn.Module):
    """
    核心扩散模型架构。
    输入原子序数与带噪声的坐标，通过多层 EGNN 预测去噪后的结构及其物理属性。
    """
    def __init__(self, num_layers=6, feat_dim=64, time_emb_dim=64):
        super(DiffusionModel, self).__init__()
        # 元素周期表 Embedding
        self.embedding = nn.Embedding(119, feat_dim) 
        
        # 时间步 Embedding
        self.time_emb = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        self.layers = nn.ModuleList([
            EGNN_Layer(feat_dim, time_emb_dim=time_emb_dim) for _ in range(num_layers)
        ])
        
        # 属性预测头
        self.her_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, 1)
        )
        self.energy_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, 1)
        )
        self.synth_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, z, pos, edge_index, t, cond=None):
        """
        前向传播
        Args:
            z: 原子序数 (torch.LongTensor)
            pos: 当前原子坐标 (torch.FloatTensor)
            edge_index: 图邻接关系
            t: 当前时间步 (torch.FloatTensor, shape [1] 或 [batch_size, 1])
        Returns:
            pred_dict: 包含去噪坐标和预测属性的字典
        """
        # 1. 嵌入原子特征和时间特征
        h = self.embedding(z)
        t_emb = self.time_emb(t.view(-1, 1).float())
        
        # 2. 消息传递层 (每一层都注入时间信息)
        orig_pos = pos
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, t_emb=t_emb)
            
        # 3. 预测噪声 (这里我们预测相对于输入坐标的位移，即噪声 epsilon)
        noise_pred = pos - orig_pos
        
        # 4. 全局池化 (聚合节点特征以预测整体属性)
        h_global = torch.mean(h, dim=0, keepdim=True)
        
        # 5. 属性预测
        her_pred = self.her_head(h_global)
        energy_pred = self.energy_head(h_global)
        synth_score = self.synth_head(h_global)
        
        return {
            'noise_pred': noise_pred, 
            'her_pred': her_pred,
            'energy_pred': energy_pred,
            'synth_score': synth_score
        }
