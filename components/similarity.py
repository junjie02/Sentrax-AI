import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import TruncatedSVD

# ===============
# 距离相关性 dCor
# ===============
def distance_correlation(x, y):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x 和 y 的样本数必须一致")

    n = x.shape[0]
    a = squareform(pdist(x, 'euclidean'))
    b = squareform(pdist(y, 'euclidean'))

    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov = np.sqrt((A * B).sum() / (n * n))
    dvar_x = np.sqrt((A * A).sum() / (n * n))
    dvar_y = np.sqrt((B * B).sum() / (n * n))

    return 0 if dvar_x * dvar_y == 0 else dcov / np.sqrt(dvar_x * dvar_y)

# ===============
# 主函数（只计算 dCor, 可选快速模式）
# ===============
def compute_dcor(hidden, logits, n_components=128, sample_tokens=256, sample_vocab=2000):
    # 去掉 batch 维度
    if hidden.ndim == 3:
        hidden = hidden.squeeze(0)
    if logits.ndim == 3:
        logits = logits.squeeze(0)

    # 转 numpy
    if hasattr(hidden, 'detach'):
        hidden = hidden.detach().cpu().numpy()
    if hasattr(logits, 'detach'):
        logits = logits.detach().cpu().numpy()

    # step0: 随机采样 token
    if hidden.shape[0] > sample_tokens:
        idx = np.random.choice(hidden.shape[0], size=sample_tokens, replace=False)
        hidden = hidden[idx]
        logits = logits[idx]

    # step1: 随机采样 vocab 维度
    if logits.shape[1] > sample_vocab:
        vocab_idx = np.random.choice(logits.shape[1], size=sample_vocab, replace=False)
        logits = logits[:, vocab_idx]

    # step2: logits 降维 (vocab_size -> n_components)
    svd_logits = TruncatedSVD(n_components=min(n_components, logits.shape[1]-1), random_state=42)
    logits_reduced = svd_logits.fit_transform(logits)  # (seq_len, n_components)

    # step3: hidden 也降维到相同维度
    svd_hidden = TruncatedSVD(n_components=min(n_components, hidden.shape[1]-1), random_state=42)
    hidden_reduced = svd_hidden.fit_transform(hidden)  # (seq_len, n_components)

    # 只计算 Distance Correlation
    dcor = distance_correlation(hidden_reduced, logits_reduced)
    return dcor

# ===============
# 使用示例
# ===============
if __name__ == "__main__":
    hidden = torch.randn(1, 1024, 1024)
    logits = torch.randn(1, 1024, 151936)

    dcor_value = compute_dcor(hidden, logits, n_components=128, sample_tokens=256, sample_vocab=2000)
    print("Approx Distance Correlation:", dcor_value)