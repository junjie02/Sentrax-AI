import torch
from trl import SFTTrainer
import torch.nn.functional as F
from components.pad import pad_logits
import yaml
import os
import math

from .weight import  teacher_confidence

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
        super(ComplexTrainer, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(model.parameters()).device
        # 输入移至设备
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        # 学生模型输出
        student_model = model.module if hasattr(model, 'module') else model

        student_outputs = student_model(** inputs)  # 输出logits
        original_loss = student_outputs.loss  # 学生原始交叉entropy损失

        # 教师模型输出（含logits和隐藏层，无梯度）
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)  # 输出logits

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
        # 1. Logits蒸馏损失计算
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
        # 2. 动态权重平均（DWA）
        # --------------------------

        #计算teacher logots 置信度
        teacher_conf = teacher_confidence(teacher_logits)
        alpha = teacher_confidence(teacher_logits)
        #print(w_kd)
        # --------------------------
        # 3. 融合所有损失
        # --------------------------
        total_loss = alpha * logits_loss + (1 - alpha) * original_loss


        if (self.state.global_step) % 100 == 0:
            self.log({
                "train/original_loss": original_loss.detach().cpu().item(),
                "train/logits_loss": logits_loss.detach().cpu().item(),
                "teacher_confidence:": teacher_conf,
            })
            
        return total_loss