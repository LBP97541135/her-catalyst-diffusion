import torch
import torch.nn as nn
import numpy as np

class GaussianDiffusion:
    def __init__(self, num_steps=100, beta_start=1e-4, beta_end=0.02, schedule_type='linear'):
        self.num_steps = num_steps
        
        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
        elif schedule_type == 'cosine':
            # 改进版余弦调度 (Improved DDPM)
            steps = num_steps + 1
            s = 0.008
            t = torch.linspace(0, num_steps, steps)
            alphas_cumprod = torch.cos(((t / num_steps) + s) / (1 + s) * np.pi / 2)**2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clamp(betas, 0, 0.999)
        else:
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
            
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # 前向加噪所需的系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 反向采样所需的系数
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def p_sample(self, model, x, t, z, edge_index):
        """
        反向采样一步: x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_theta) + sigma_t * z
        """
        device = x.device
        with torch.no_grad():
            pred_dict = model(z, x, edge_index, t)
            eps_theta = pred_dict['noise_pred']
            
        alpha_t = self.alphas[t].view(-1, 1)
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1)
        beta_t = self.betas[t].view(-1, 1)
        
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t) * eps_theta
        )
        
        if t == 0:
            return mean
        else:
            variance = self.posterior_variance[t].view(-1, 1)
            noise = torch.randn_like(x)
            return mean + torch.sqrt(variance) * noise

    def q_sample(self, x_0, t, noise=None):
        """
        前向加噪过程: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def get_params(self, device):
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self
