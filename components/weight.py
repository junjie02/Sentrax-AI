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

    # combine
    d_teacher = 0.5 * H_norm + 0.5 * M_norm
    return float(d_teacher.clamp(0, 1).item()) 

# ==============
# 隐藏层复杂度
# ==============
def hidden_complexity(hiddens, attn_maps=None, answer_mask=None, use_path=True, use_attn=True):
    """
    计算教师隐藏层的复杂度指标
    参数:
        hiddens: list of torch.Tensor, 每个是 [T, d]，选定层的hidden states
        attn_maps: list of torch.Tensor, 每个是 [H, T, T] 注意力矩阵，可选
        answer_mask: torch.BoolTensor [T]，答案相关位置为1；若None则全序列
        use_path: 是否启用路径长度指标
        use_attn: 是否启用注意力熵指标
    返回:
        c_hidden: float ∈ [0,1] 隐藏层复杂度
    """
    device = hiddens[0].device
    batch_size, T, d = hiddens[0].shape
    if answer_mask is None:
        mask = torch.ones(batch_size, T, device=device)
    else:
        mask = answer_mask.float().to(device)
    mask_norm = mask.sum().clamp_min(1.0)

    # 平均token表征（答案token）
    hs = [(h * mask.unsqueeze(-1)).sum(dim=0) / mask_norm for h in hiddens]  # [d]
    hs = torch.stack(hs, dim=0)  # [L_sel, d]

    comps, weights, wsum = [], [], 0.0

    # (a) path length
    if use_path and hs.size(0) >= 2:
        diffs = (hs[1:] - hs[:-1]).norm(dim=-1)  # [L-1]
        c_path = diffs.mean() / math.sqrt(d)
        comps.append(c_path); weights.append(0.5); wsum += 0.5

    # (b) attention entropy
    if use_attn and attn_maps is not None:
        entropies = []
        for A in attn_maps:  # [H, T, T]
            q_mask = answer_mask if answer_mask is not None else torch.ones(T, dtype=torch.bool, device=device)
            A_q = A[:, q_mask, :]  # [H, T_ans, T]
            A_q = A_q / (A_q.sum(dim=-1, keepdim=True).clamp_min(1e-9))
            ent = -(A_q * (A_q.clamp_min(1e-9)).log()).sum(dim=-1) / math.log(T)
            entropies.append(ent.mean())
        c_attn = torch.stack(entropies).mean()
        comps.append(c_attn); weights.append(0.5); wsum += 0.5

    if wsum > 0:
        c_hidden = sum(w*c for w, c in zip(weights, comps)) / wsum
    else:
        c_hidden = torch.tensor(0.0, device=device)

    return float(c_hidden.clamp(0, 1).item()) * 100

# ==============
# 隐藏层权重计算
# ==============
def hidden_weight(d_teacher, c_hidden, t_norm,
                  alpha=0.5, mu=0.55, sigma=0.25,
                  gamma=0.4, lamb=3.0, wmin=0.1, wmax=0.9):
    """
    综合难度 + 钟形函数 + 训练阶段门控
    参数:
        d_teacher: float ∈ [0,1] 教师置信度难度
        c_hidden: float ∈ [0,1] 隐藏层复杂度
        t_norm: float ∈ [0,1] 训练进度（当前epoch / 总epoch）
        alpha: 权重，混合teacher与hidden (默认0.5)
        mu: 钟形曲线中心，代表“最注重过程”的难度 (默认0.6)
        sigma: 钟形曲线宽度 (默认0.25)
        gamma: 训练后期最小保持过程权重比例 (默认0.4)
        lamb: 训练进度衰减速度 (默认3.0)
        wmin, wmax: 权重裁剪上下限
    返回:
        w_hidden: float, 隐藏层loss的权重
        d: float, 综合难度
    """
    d = d_teacher * c_hidden 
    bell = math.exp(- (d - mu) ** 2 / (2 * sigma ** 2))
    stage = gamma + (1 - gamma) * math.exp(-lamb * t_norm)
    w = stage * bell
    w = max(min(w, wmax), wmin)
    #print(f"d = {d}")
    return w, d

# =====================
# 使用示例
# =====================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 模拟教师输出
    T, V, d = 128, 5000, 768
    teacher_logits = torch.randn(T, V, device=device)
    teacher_hiddens = [torch.randn(T, d, device=device) for _ in range(4)]  # 模拟4层
    teacher_attn = [torch.rand(8, T, T, device=device) for _ in range(2)]   # 模拟2层attention

    answer_mask = torch.zeros(T, dtype=torch.bool, device=device)
    answer_mask[-10:] = 1  # 模拟最后10个token是答案

    # Step 1: 教师置信度
    d_teacher = teacher_confidence(teacher_logits, answer_mask)
    # Step 2: 隐藏层复杂度
    c_hidden = hidden_complexity(teacher_hiddens, teacher_attn, answer_mask)
    # Step 3: 权重计算（假设训练在50%进度）
    w_hidden, d = hidden_weight(d_teacher, c_hidden, t_norm=0.5)

    print(f"教师难度 d_teacher={d_teacher:.3f}, 隐藏复杂度 c_hidden={c_hidden:.3f}")
    print(f"综合难度 d={d:.3f}, 最终隐藏层权重 w_hidden={w_hidden:.3f}")
