import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback
import yaml
from accelerate import Accelerator
import math

# -------------------- 配置参数 设置环境--------------------
config = {
    "project_name": "distil-logits",
    "dataset": {
        "name": r"C:\Users\15746\Desktop\fsdownload\cybersecurity_alarm_analysis",  # 数据集路径（需包含instruction/input/output字段）
        "split": "train",
        "seed": 42,
        "num_samples": 100
    },
    "models": {
        "teacher": r"C:\Users\15746\Desktop\fsdownload\Qwen3-4B",  # 教师模型
        "student": r"C:\Users\15746\Desktop\fsdownload\Qwen3-0.6B"  # 学生模型
    },
    "tokenizer": {
        "max_length": 4096,
        "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    },
    "training": {
        "output_dir": "./results_alpaca",
        "num_train_epochs": 6,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "save_steps": 1000,
        "logging_steps": 1,
        "learning_rate": 1e-5,
        "weight_decay": 0.05,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "resume_from_checkpoint": None,
        "fp16": False,
        "bf16": True,
        "remove_unused_columns": False,
        "max_grad_norm": 30,
        "eval_strategy": "steps",
        "eval_steps": 1000,
        "per_device_eval_batch_size": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False
    },
    "distillation": {
        "temperature": 2.0,
        "alpha": 0.5,  # 蒸馏损失与原始损失的权衡系数
        "use_output_cross_entropy": True,  # 是否用output计算交叉熵损失（建议True，约束answer精准性）
        "hidden_loss_scale": 0.005,
        "top_k": 4
    },
    "model_config": {
        "use_flash_attention": False
    },
    #"spectrum": {
    #    "layers_to_unfreeze": "/mnt/sda1/Yikun/distillKitPlus/spectrum/snr_results_-mnt-sda1-Yikun-distillKitPlus-Qwen3-0.6B-_unfrozenparameters_50percent.yaml"
    #}
}

# os.environ['WANDB_MODE'] = 'offline'
# os.environ['WANDB_PROJECT'] = config["project_name"]
# os.environ["WANDB_DIR"] = "/mnt/sda1/Yikun/distillKitPlus/DistillKit/wandb/"

accelerator = Accelerator()
device = accelerator.device


# -------------------- 数据集加载与处理 --------------------
def prepare_dataset(example):
    """
    处理数据集：拼接instruction+input作为模型输入，output作为目标输出
    格式要求：数据集需包含instruction（分析任务）、input（日志内容）、output（参考结果）字段
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")
    
    # 构造模型输入（prompt）和目标输出（response）
    model_input = f"{instruction}\n{input_text}"
    model_output = output_text  # 如："<reasoning>...</reasoning>\n<answer>攻击/误报</answer>"
    
    # 用tokenizer处理成模型可训练的格式（对齐chat_template）
    messages = [
        {"role": "system", "content": "You are a network security alert analysis expert."},
        {"role": "user", "content": model_input},
        {"role": "assistant", "content": model_output}
    ]
    
    # 分别用教师和学生的tokenizer处理（确保格式一致）
    def tokenize_with_template(tokenizer, messages):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer(text, truncation=True, max_length=config["tokenizer"]["max_length"], padding="max_length")
    
    student_encodings = tokenize_with_template(student_tokenizer, messages)
    teacher_encodings = tokenize_with_template(teacher_tokenizer, messages)
    
    output_encodings = student_tokenizer(
    output_text, 
    truncation=True, 
    max_length=config["tokenizer"]["max_length"], 
    padding="max_length"
)
    
    result = {
        "input_ids": student_encodings["input_ids"],
        "attention_mask": student_encodings["attention_mask"],
        "labels": output_encodings["input_ids"],
        # 将教师字段作为嵌套字典存入一个新字段（如"teacher_data"）
        "teacher_data": {
            "input_ids": teacher_encodings["input_ids"],
            "attention_mask": teacher_encodings["attention_mask"]
        }
    }
    return result


# -------------------- Tokenizer 加载 --------------------
teacher_tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"])
student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])
student_tokenizer.chat_template = config["tokenizer"]["chat_template"]

# 加载数据集并打乱
dataset = load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])
dataset = dataset.shuffle(seed=config["dataset"]["seed"])

#采样
if "num_samples" in config["dataset"]:
    dataset = dataset.select(range(config["dataset"]["num_samples"]))

# 预处理数据集，数据集划分
print("Preprocessing and tokenizing dataset...")
original_columns = dataset.column_names
dataset = dataset.map(prepare_dataset, remove_columns=original_columns)
dataset = dataset.train_test_split(test_size=0.1, seed=config["dataset"]["seed"])

# 以训练集第1个样本为例，将 labels 的 token ID 转回文本
sample_labels_ids = dataset["train"][0]["labels"]

# 用 student_tokenizer 解码（排除特殊token，如 [PAD] [CLS]）
decoded_labels = student_tokenizer.decode(
    sample_labels_ids,
    skip_special_tokens=True  # 跳过 [CLS] [SEP] [PAD] 等特殊token
)

print("训练集第1个样本 input_ids 解码后：")
print(decoded_labels)  # 输出人类可读的文本标签

# 处理完数据集后添加
print("Dataset sample keys:", dataset["train"][0].keys())


# -------------------- 模型加载 --------------------
model_kwargs = {"torch_dtype": torch.bfloat16}
if config["model_config"]["use_flash_attention"]:
    model_kwargs["attn_implementation"] = "flash_attention_2"

teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"], **model_kwargs).to(device)
student_model = AutoModelForCausalLM.from_pretrained(config["models"]["student"], **model_kwargs).to(device)

#冻结参数callback
class FreezeCallback(TrainerCallback):
    def __init__(self, student_model, unfrozen_layers_file):
        self.student_model = student_model
        self.unfrozen_layers_file = unfrozen_layers_file
        # 加载需要解冻的层（spectrum配置）
        with open(unfrozen_layers_file, 'r') as file:
            self.unfrozen_layers = yaml.safe_load(file)['unfrozen_parameters']
        # 初始状态：所有参数可训练（前4个epoch）

    def set_spectrum_freeze(self):
        """根据spectrum配置冻结参数（仅保留指定层可训练）"""
        for name, param in self.student_model.named_parameters():
            if any(layer in name for layer in self.unfrozen_layers):
                param.requires_grad = True  # 解冻指定层
            else:
                param.requires_grad = False  # 冻结其他层

    def on_epoch_begin(self, args, state, control,** kwargs):
        # state.epoch是从0开始的浮点数（如3.0表示第4个epoch开始）
        current_epoch = int(state.epoch)  # 转换为整数（0→第1个epoch，4→第5个epoch）
        
        if current_epoch < 1:  # 前4个epoch（0-3对应第1-4个epoch）：全参数训练
            #不冻结任何参数
            if int(state.epoch) == 0:  # 只在首次epoch开始时打印一次
                print("前1个epoch：所有参数可训练（不冻结）")
        else:  # 第5个epoch及以后：启用spectrum冻结
            self.set_spectrum_freeze()
            print(f"第{current_epoch+1}个epoch：启用spectrum参数冻结逻辑")

#将学生模型的隐藏层适配到教师模型的隐藏层
class MultiLayerAdaptationLayer(torch.nn.Module):
    def __init__(self, student_dim, teacher_dim, num_student_layers, num_teacher_layers, top_k=4, dtype=torch.bfloat16):
        super().__init__()
        self.projections = torch.nn.ModuleList([#创建多层线性投影
            torch.nn.Linear(student_dim, teacher_dim, dtype=dtype)
            for _ in range(num_student_layers)
        ])
        self.layer_mapping = self.create_layer_mapping(num_student_layers, num_teacher_layers, top_k)
        self.dtype = dtype

    def create_layer_mapping(self, num_student_layers, num_teacher_layers, top_k):#具体映射层
        mapping = {}
        for i in range(num_student_layers - top_k, num_student_layers):
            j = num_teacher_layers - top_k + i - (num_student_layers - top_k)
            if j < num_teacher_layers:
                mapping[i] = j
        return mapping

    def forward(self, student_hidden_states):#创建适配层
        adapted_hidden_states = []
        for i, hidden_state in enumerate(student_hidden_states):
            if i not in self.layer_mapping:
                continue  # 跳过未映射的层
            adapted_hidden_states.append(self.projections[i](hidden_state.to(self.dtype)))
        return adapted_hidden_states

adaptation_layer = MultiLayerAdaptationLayer(#创建适配层实例，移动到gpu
    student_model.config.hidden_size,
    teacher_model.config.hidden_size,
    student_model.config.num_hidden_layers,
    teacher_model.config.num_hidden_layers,
    dtype=torch.bfloat16,
    top_k=config["distillation"]["top_k"]
).to(device)

print("Student <-> Teacher layer mapping:")
for student_idx, teacher_idx in adaptation_layer.layer_mapping.items():
    print(f"Student layer {student_idx}  --> Teacher layer {teacher_idx}")


def pad_logits(student_logits, teacher_logits):#将老师和学生的logits进行填充
    student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)
    if student_size != teacher_size:
        pad_size = abs(student_size - teacher_size)
        pad_tensor = torch.zeros((*teacher_logits.shape[:-1], pad_size), dtype=teacher_logits.dtype, device=teacher_logits.device)
        return (torch.cat([student_logits, pad_tensor], dim=-1), teacher_logits) if student_size < teacher_size else (student_logits, torch.cat([teacher_logits, pad_tensor], dim=-1))
    return student_logits, teacher_logits


# -------------------- 自定义Trainer（核心蒸馏逻辑） --------------------
class DistillTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.teacher_model = kwargs.pop("teacher_model")
        self.adaptation_layer = kwargs.pop("adaptation_layer")
        self.use_output_cross_entropy = config["distillation"]["use_output_cross_entropy"]
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 设备对齐
        inputs = {k: v.to(device) for k, v in inputs.items()}
        teacher_model = self.teacher_model.to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

        # 学生模型输出（含logits、隐藏层）
        student_outputs = model(**inputs, output_hidden_states=True)
        student_logits = student_outputs.logits
        student_hiddens = student_outputs.hidden_states

        #输出实验*********************************************************************************************
        # ---------- 调试学生输出 ----------
        # pred_ids = student_logits.argmax(dim=-1)  # 取最大概率的 token id
        # texts = [student_tokenizer.decode(ids, skip_special_tokens=True) for ids in pred_ids]

        # print("\n[Student Output Debug]")
        # for i, t in enumerate(texts[:2]):  # 只打印前两个，避免刷屏
        #     print(f"Sample {i}: {t}")
# --------------------------------
        #输出实验*********************************************************************************************
        
        # 教师模型输出（无梯度）
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs, output_hidden_states=True)
            teacher_logits = teacher_outputs.logits
            teacher_hiddens = teacher_outputs.hidden_states

        # 1. Logits蒸馏损失（原有逻辑）
        student_logits_scaled = student_logits / config["distillation"]["temperature"]
        teacher_logits_scaled = teacher_logits / config["distillation"]["temperature"]
        logits_loss = F.kl_div(
            F.log_softmax(student_logits_scaled, dim=-1),
            F.softmax(teacher_logits_scaled, dim=-1),
            reduction='batchmean'
        ) * (config["distillation"]["temperature"] ** 2) / config["tokenizer"]["max_length"]

        # 2. 隐藏层蒸馏损失（原有逻辑）
        adapted_student_hiddens = self.adaptation_layer(student_hiddens)
        hidden_loss = torch.tensor(0.0, device=device)
        matched = 0
        for list_idx, (student_idx, teacher_idx) in enumerate(self.adaptation_layer.layer_mapping.items()):
            s_hidden = adapted_student_hiddens[list_idx]
            t_hidden = teacher_hiddens[teacher_idx]
            if s_hidden.shape == t_hidden.shape:
                hidden_loss += F.mse_loss(s_hidden, t_hidden)
                matched += 1
        avg_hidden_loss = hidden_loss / max(1, matched) if matched > 0 else 0.0

        # 4. 总损失融合（蒸馏损失 + 交叉熵损失）
        total_loss =  (logits_loss + 0.004 * avg_hidden_loss)

        # 日志记录（可选）
        if self.state.global_step % 100 == 0:
            self.log({
                "train/logits_loss": logits_loss.item(),
                "train/hidden_loss": avg_hidden_loss.item() * 0.004,
                "train/total_loss": total_loss.item()
            })

        return (total_loss, student_outputs) if return_outputs else total_loss
    
# -------------------- 训练参数与Trainer初始化 --------------------
# 替换TrainingArguments为SFTConfig，并显式指定保留的字段
sft_config = SFTConfig(
    **config["training"],
    dataset_text_field=None,  # 不需要文本字段（已预处理为input_ids）
    dataset_kwargs={"keep_in_memory": True},
)

trainer = DistillTrainer(
    model=student_model,
    teacher_model=teacher_model,
    adaptation_layer=adaptation_layer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=sft_config,  # 使用SFTConfig而非TrainingArguments
)

# 添加Spectrum冻结回调（若配置存在）
if "spectrum" in config and "layers_to_unfreeze" in config["spectrum"]:
    freeze_callback = FreezeCallback(student_model, config["spectrum"]["layers_to_unfreeze"])
    trainer.add_callback(freeze_callback)

# -------------------- 训练与保存 --------------------
trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])
trainer.save_model(config["training"]["output_dir"])