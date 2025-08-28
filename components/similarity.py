import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SmallMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def compute_nn_r2(hidden, logits, sample_tokens=256, sample_vocab=2000, hidden_mlp=128, epochs=20, lr=1e-3, device='cuda'):
    """
    用小 MLP 回归 logits -> R² 作为相关性指标

    参数:
        hidden: torch.Tensor, [1, seq_len, hidden_dim]
        logits: torch.Tensor, [1, seq_len, vocab_size]
        sample_tokens: 随机采样 token 数
        sample_vocab: 随机采样 vocab 数
        hidden_mlp: MLP 隐藏层大小
        epochs: 训练轮数
        lr: 学习率
        device: 'cpu' or 'cuda'

    返回:
        r2: float, 回归 R² 值
    """
    seq_len, hidden_dim = hidden.shape[1], hidden.shape[2]
    vocab_size = logits.shape[2]

    # 随机采样
    token_idx = torch.randperm(seq_len)[:sample_tokens]
    vocab_idx = torch.randperm(vocab_size)[:sample_vocab]

    hidden_sample = hidden[0, token_idx].detach().to(device)       # [sample_tokens, hidden_dim]
    logits_sample = logits[0, token_idx][:, vocab_idx].detach().to(device)  # [sample_tokens, sample_vocab]

    # 建立模型
    mlp = SmallMLP(hidden_dim, sample_vocab, hidden_mlp).to(device)
    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 训练 MLP
    mlp.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = mlp(hidden_sample)
        loss = criterion(pred, logits_sample)
        loss.backward()
        optimizer.step()

    # 计算 R²
    mlp.eval()
    with torch.no_grad():
        pred = mlp(hidden_sample)
        ss_res = ((logits_sample - pred) ** 2).sum()
        ss_tot = ((logits_sample - logits_sample.mean(0)) ** 2).sum()
        r2 = 1 - (ss_res / (ss_tot + 1e-8)).item()

    return r2

# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden = torch.randn(1, 1024, 1024, requires_grad=True).to(device)
    logits = torch.randn(1, 1024, 151936, requires_grad=True).to(device)

    r2_score = compute_nn_r2(hidden, logits, sample_tokens=256, sample_vocab=2000, hidden_mlp=128, epochs=20, device=device)
    print("NN Regression R²:", r2_score)