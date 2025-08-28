import torch
import numpy as np

def pdist(x):
    """计算 pairwise 欧式距离"""
    r = torch.sum(x * x, 1).reshape(-1, 1)
    dist = r - 2 * torch.matmul(x, x.t()) + r.t()
    dist = torch.sqrt(torch.clamp(dist, min=0))
    return dist

def dcor(x, y):
    """计算 Distance Correlation"""
    n = x.shape[0]
    a = pdist(x)
    b = pdist(y)

    A = a - a.mean(0, keepdim=True) - a.mean(1, keepdim=True) + a.mean()
    B = b - b.mean(0, keepdim=True) - b.mean(1, keepdim=True) + b.mean()

    dcov2_xy = (A * B).sum() / (n * n)
    dcov2_xx = (A * A).sum() / (n * n)
    dcov2_yy = (B * B).sum() / (n * n)

    dcor = dcov2_xy / torch.sqrt(dcov2_xx * dcov2_yy + 1e-10)
    return dcor

def compute_dcor(hidden, logits, sample_tokens=256, sample_vocab=2000, device='cuda'):
    """
    用 Distance Correlation 衡量 hidden 与 logits 的相关性

    参数:
        hidden: torch.Tensor, [1, seq_len, hidden_dim]
        logits: torch.Tensor, [1, seq_len, vocab_size]
        sample_tokens: 随机采样 token 数
        sample_vocab: 随机采样 vocab 数
        device: 'cpu' or 'cuda'

    返回:
        dcor: float, distance correlation 值
    """
    seq_len, hidden_dim = hidden.shape[1], hidden.shape[2]
    vocab_size = logits.shape[2]

    # 随机采样
    token_idx = torch.randperm(seq_len)[:sample_tokens]
    vocab_idx = torch.randperm(vocab_size)[:sample_vocab]

    hidden_sample = hidden[0, token_idx].detach().to(device)       # [sample_tokens, hidden_dim]
    logits_sample = logits[0, token_idx][:, vocab_idx].detach().to(device)  # [sample_tokens, sample_vocab]

    # 计算 dCor
    return dcor(hidden_sample, logits_sample).item()

# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden = torch.randn(1, 1024, 1024).to(device)
    logits = torch.randn(1, 1024, 151936).to(device)

    dcor_score = compute_dcor(hidden, logits, sample_tokens=256, sample_vocab=2000, device=device)
    print("Distance Correlation:", dcor_score)
