import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============
# 教师 logits 置信度
# ==============
def teacher_confidence(logits, answer_mask=None):
    """
    计算教师 logits 的置信度难度指标
    参数:
        logits: torch.Tensor [T, V]  教师输出的logits（单个样本）
        answer_mask: torch.BoolTensor [T], 只在答案相关位置为1；若None则全序列计算
    返回:
        d_teacher: float ∈ [0,1]  教师置信度难度
    """
    device = logits.device
    #print(logits.shape)
    batch_size, T, V = logits.shape
    probs = F.softmax(logits, dim=-1)

    if answer_mask is None:
        mask = torch.ones(batch_size, T, dtype=torch.float, device=device)
    else:
        mask = answer_mask.float().to(device)
    T_eff = mask.sum().clamp_min(1.0)

    # (a) Normalized entropy
    entropy = -(probs * (probs.clamp_min(1e-9)).log()).sum(dim=-1)  # [T]
    H = (entropy * mask).sum() / T_eff
    H_norm = H / math.log(V)

    # (b) Top-2 margin
    top2 = probs.topk(k=2, dim=-1).values  # [batch_size,T,2]
    margin = (top2[..., 0] - top2[..., 1]).clamp_min(0)  # [T]
    m = (margin * mask).sum() / T_eff
    M_norm = 1.0 - m  # 高=难
    
    # (c) Top-k mass （前k项概率和，越大说明前几项都很有信心）
    topk = probs.topk(k=5, dim=-1).values  # [B,T,k]
    topk_sum = topk.sum(dim=-1)  # [B,T]
    C = (topk_sum * mask).sum() / T_eff
    C_norm = C.clamp(0, 1)

    # combine
    d_teacher = 0.4 * H_norm + 0.3 * M_norm + 0.3 * C_norm
    d_teacher = sigmoid(d_teacher)  # 非线性映射
    return float(d_teacher.clamp(0, 1).item()) 


def sigmoid(x, k=15, x0=0.5):
    return 1 / (1 + torch.exp(-k * (x - x0)))
