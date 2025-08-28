import torch
from transformers import Trainer as SFTTrainer
import torch.nn.functional as F
from components.pad import pad_logits
import yaml
import os
import math

from .similarity import compute_dcor

#加载配置文件
CONFIG_PATH = "././config.yaml"
# 读取YAML文件并加载配置
def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)  # safe_load避免加载恶意代码
    return config

# 执行加载
config = load_config(CONFIG_PATH)

class ComplexTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.remove_unused_columns = kwargs.pop('remove_unused_columns', None)
        self.teacher_model = kwargs.pop("teacher_model")  # 显式接收教师模型
        self.adaptation_layer = kwargs.pop("adaptation_layer")  # 显式接收适配层
        self.T = kwargs.pop("dwa_temperature", config["distillation"]["dwa_temperature"])  # DWA 温度参数
        #self.max_seq_length = kwargs.get('max_seq_length', 1024)
        super(ComplexTrainer, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(model.parameters()).device
        # 输入移至设备
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        # self.teacher_model = self.teacher_model.to(device)  

        #print(inputs)
        # 冻结教师模型参数
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

        self.adaptation_layer = self.adaptation_layer.to(device)

        # 学生模型输出（含logits和隐藏层）
        student_model = model.module if hasattr(model, 'module') else model
        student_outputs = student_model(** inputs, output_hidden_states=True)  # 输出logits和隐藏层
        original_loss = student_outputs.loss  # 学生原始交叉entropy损失

        # 教师模型输出（含logits和隐藏层，无梯度）
        with torch.no_grad():
            teacher_inputs = inputs.copy()
            teacher_inputs['input_ids'] = inputs['teacher_input_ids']
            teacher_inputs['attention_mask'] = inputs['teacher_attention_mask']
            teacher_outputs = self.teacher_model(**teacher_inputs, output_hidden_states=True)  # 输出logits和隐藏层

        # 计算混合蒸馏损失
        hybrid_loss = self.hybrid_distillation_loss(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            original_loss=original_loss,
            device=device
        )
        return (hybrid_loss, student_outputs) if return_outputs else hybrid_loss

    def hybrid_distillation_loss(self, student_outputs, teacher_outputs, original_loss, device):
        # --------------------------
        # 1. 隐藏层蒸馏损失计算
        # --------------------------
        student_hiddens = student_outputs.hidden_states  # 学生隐藏层（tuple，包含各层输出）
        teacher_hiddens = teacher_outputs.hidden_states  # 教师隐藏层

        # 适配学生隐藏层维度
        adapted_student_hiddens = self.adaptation_layer(student_hiddens)

        hidden_loss = torch.tensor(0.0, device=device)
        matched = 0
        # 遍历映射的层对，计算损失
        for list_idx, (student_idx, teacher_idx) in enumerate(self.adaptation_layer.layer_mapping.items()):

            s_hidden = adapted_student_hiddens[list_idx]
            t_hidden = teacher_hiddens[teacher_idx]

            # 确保形状匹配
            if s_hidden.shape != t_hidden.shape:
                raise ValueError(f"Hidden shape mismatch: student {s_hidden.shape} vs teacher {t_hidden.shape}")

            # 用MSE散度计算隐藏层分布差异
            layer_loss = F.mse_loss(s_hidden, t_hidden) 
            hidden_loss += layer_loss
            matched += 1
        # 平均隐藏层损失（除以层数）
        avg_hidden_loss = hidden_loss / max(1, matched)

        # --------------------------
        # 2. Logits蒸馏损失计算
        # --------------------------
        student_logits, teacher_logits = pad_logits(
            student_outputs.logits.to(device),
            teacher_outputs.logits.to(device)
        )
        temp = config["distillation"]["temperature"]
        s_logits_scaled = student_logits / temp
        t_logits_scaled = teacher_logits / temp
        logits_loss = F.kl_div(
            F.log_softmax(s_logits_scaled, dim=-1),
            F.softmax(t_logits_scaled, dim=-1),
            reduction='batchmean'
        ) * (temp** 2) / config["tokenizer"]["max_length"]

        # --------------------------
        # 3. 动态权重平均（DWA）
        # --------------------------
        student_last_hidden = student_outputs.hidden_states[-1]  # 取最后一层 hidden
        teacher_last_hidden = teacher_outputs.hidden_states[-1]
        student_hidden_logits_similarity = (compute_dcor(student_last_hidden, student_logits) - 0.92)*10
        teacher_hidden_logits_similarity =  compute_dcor(teacher_last_hidden, teacher_logits)

        w_kd = student_hidden_logits_similarity * teacher_hidden_logits_similarity
        # --------------------------
        # 4. 融合所有损失
        # --------------------------
        # 蒸馏损失 = 隐藏层损失 * hidden_weight + logits损失 * (1 - hidden_weight)
        # kd_loss = (config["distillation"]["hidden_weight"] * avg_hidden_loss +
        #           (1 - config["distillation"]["hidden_weight"]) * logits_loss)
        #w_kd = max(w_kd, 0.6)
        total_loss = w_kd * logits_loss + (1-w_kd) * (avg_hidden_loss * config["distillation"]["hidden_loss_scale"])
        # 总损失 = 蒸馏损失 * alpha + 原始损失 * (1 - alpha)
        total_loss = config["distillation"]["alpha"] * total_loss + (1 - config["distillation"]["alpha"]) * original_loss

        if (self.state.global_step) % 100 == 0:
            self.log({
                "train/original_loss": original_loss.detach().cpu().item(),
                "train/logits_loss": logits_loss.detach().cpu().item(),
                "train/hidden_loss": avg_hidden_loss.detach().cpu().item(),
                "w_kd": w_kd,
            })


        return total_loss