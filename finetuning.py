import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback
import yaml
from accelerate import Accelerator
import math


# Configuration
config = {
    "project_name": "finetuning",
    "dataset": {
        "name": "/mnt/sda1/Yikun/distillKitPlus/cybersecurity_alarm_analysis/",
        "split": "train",
        "num_samples": 50000, # You can pass a number here to limit the number of samples to use.
        "seed": 42
    },
    "models": {
        "teacher": "/mnt/sda1/Yikun/distillKitPlus/Qwen3-4B-Instruct-2507/",
        "student": "/mnt/sda1/Yikun/distillKitPlus/DistillKit/results_en/"
    },
    "tokenizer": {
        "max_length": 6000,
        "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    },
    "training": {
        "output_dir": "./results_finetuning",
        "num_train_epochs": 5,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "save_steps": 1000,
        "logging_steps": 1,
        "learning_rate": 1e-5,
        "weight_decay": 0.05,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "resume_from_checkpoint": None,  # Set to a path or True to resume from the latest checkpoint
        "fp16": False,
        "bf16": True,
        "remove_unused_columns": False,
        "max_grad_norm": 8,
        "eval_strategy": "steps",  # 按步骤评估
        "eval_steps": 1000,  # 每500步评估一次
        "per_device_eval_batch_size": 1,  # 评估时的batch size
        "load_best_model_at_end": True,  # 训练结束时加载最佳模型
        "metric_for_best_model": "eval_loss",  # 以评估损失作为最佳模型的指标
        "greater_is_better": False  # 损失越小越好
        #"report_to": "wandb"
    },
    "distillation": {
        "temperature": 2.0,
        "alpha": 0.0,
        "dwa_temperature": 2.0,  # 新增：DWA温度参数
        "hidden_loss_scale": 0.015, # 新增：隐藏层损失缩放系数（MSE）
        "top_k": 4  # 新增：适配层映射的top-k层数
    },
    "model_config": {
        "use_flash_attention": True
    },
     "spectrum": {
         "layers_to_unfreeze":"/mnt/sda1/Yikun/distillKitPlus/spectrum/snr_results_-mnt-sda1-Yikun-distillKitPlus-Qwen3-0.6B-_unfrozenparameters_50percent.yaml" # You can pass a spectrum yaml file here to freeze layers identified by spectrum.
     }
}

# 设置wandb 
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_PROJECT'] = config["project_name"]
os.environ["WANDB_DIR"] = "/mnt/sda1/Yikun/distillKitPlus/DistillKit/wandb/"

accelerator = Accelerator()
device = accelerator.device

# 加载数据集
dataset = load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])
dataset = dataset.shuffle(seed=config["dataset"]["seed"])
if "num_samples" in config["dataset"]:
    dataset = dataset.select(range(config["dataset"]["num_samples"]))

# Load tokenizers
teacher_tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"])
student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])

# Apply chat template to student tokenizer
student_tokenizer.chat_template = config["tokenizer"]["chat_template"]


#sharegpt格式数据集处理
def prepare_dataset(example):
    conversations = example['conversations']
    message = []
    
    if isinstance(conversations, list):
        for conversation in conversations:
            if isinstance(conversation, dict):
                if conversation.get('from') == 'human':
                    message.append({"role": "user", "content": conversation.get('value', '')})
                elif conversation.get('from') == 'gpt':
                    message.append({"role": "assistant", "content": conversation.get('value', '')})
                elif conversation.get('from') == 'system':
                    message.insert(0, {"role": "system", "content": conversation.get('value', '')})

    if not any(msg.get('role') == 'system' for msg in message):
        message.insert(0, {"role": "system", "content": "You are a helpful assistant."})
    
    student_text = student_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    teacher_text = teacher_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    
    student_encodings = student_tokenizer(student_text, truncation=True, max_length=config["tokenizer"]["max_length"], padding='max_length')
    teacher_encodings = teacher_tokenizer(teacher_text, truncation=True, max_length=config["tokenizer"]["max_length"], padding='max_length')

    return {
        "input_ids": student_encodings["input_ids"],
        "attention_mask": student_encodings["attention_mask"],
        "teacher_input_ids": teacher_encodings["input_ids"],
        "teacher_attention_mask": teacher_encodings["attention_mask"],
    }


# 预处理数据集，数据集划分
print("Preprocessing and tokenizing dataset...")
original_columns = dataset.column_names
dataset = dataset.map(prepare_dataset, remove_columns=original_columns, num_proc=8)
dataset = dataset.train_test_split(test_size=0.1, seed=config["dataset"]["seed"])

print("Dataset preparation complete. Loading models...")

# Load models with configurable flash attention
model_kwargs = {"torch_dtype": torch.bfloat16}
if config["model_config"]["use_flash_attention"]:
    model_kwargs["attn_implementation"] = "flash_attention_2"

teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"], **model_kwargs).to(accelerator.device)
student_model = AutoModelForCausalLM.from_pretrained(config["models"]["student"], **model_kwargs).to(accelerator.device)


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


# Optionally freeze layers of the student model based on spectrum configuration
# if "spectrum" in config and "layers_to_unfreeze" in config["spectrum"]:
#     print("using spectrum unfrozen.........")
#     def freeze_student_spectrum(model, unfrozen_layers_file):
#         with open(unfrozen_layers_file, 'r') as file:
#             unfrozen_layers = yaml.safe_load(file)['unfrozen_parameters']
        
#         for name, param in model.named_parameters():
#             if not any(layer in name for layer in unfrozen_layers):
#                 param.requires_grad = False
#             else:
#                 param.requires_grad = True

    # Apply freezing to student model
#     freeze_student_spectrum(student_model, config["spectrum"]["layers_to_unfreeze"])
# else:
#     print("Spectrum configuration not found. All layers of the student model will be trainable.")


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

class ComplexTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.remove_unused_columns = kwargs.pop('remove_unused_columns', None)
        self.teacher_model = kwargs.pop("teacher_model")  # 显式接收教师模型
        self.adaptation_layer = kwargs.pop("adaptation_layer")  # 显式接收适配层
        self.T = kwargs.pop("dwa_temperature", config["distillation"]["dwa_temperature"])  # DWA 温度参数
        #self.max_seq_length = kwargs.get('max_seq_length', 1024)
        self.loss_history = {
            "kd_loss": [],
            "hidden_loss": []
        }
        super(ComplexTrainer, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(model.parameters()).device
        # 输入移至设备
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        # self.teacher_model = self.teacher_model.to(device)  

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
            teacher_outputs = self.teacher_model(**inputs, output_hidden_states=True)  # 输出logits和隐藏层

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
        self.loss_history["kd_loss"].append(logits_loss.item())
        self.loss_history["kd_loss"] = self.loss_history["kd_loss"][-5:]  # 保持最近20个记录
        self.loss_history["hidden_loss"].append(avg_hidden_loss.item())
        self.loss_history["hidden_loss"] = self.loss_history["hidden_loss"][-5:]

        w_kd, w_hidden = 1.0, 1.0  # 默认权重
        if len(self.loss_history["kd_loss"]) >= 2:
            r_kd = self.loss_history["kd_loss"][-1] / (self.loss_history["kd_loss"][-2] + 1e-8)
            r_hidden = self.loss_history["hidden_loss"][-1] / (self.loss_history["hidden_loss"][-2] + 1e-8)

            exp_kd = math.exp(r_kd / self.T)
            exp_hidden = math.exp(r_hidden / self.T)
            sum_exp = exp_kd + exp_hidden
            w_kd = exp_kd / sum_exp
            w_hidden = exp_hidden / sum_exp
        # --------------------------
        # 4. 融合所有损失
        # --------------------------
        # 蒸馏损失 = 隐藏层损失 * hidden_weight + logits损失 * (1 - hidden_weight)
        # kd_loss = (config["distillation"]["hidden_weight"] * avg_hidden_loss +
        #           (1 - config["distillation"]["hidden_weight"]) * logits_loss)
        
        total_loss = w_kd * logits_loss + (1-w_kd) * (avg_hidden_loss * config["distillation"]["hidden_loss_scale"])
        # 总损失 = 蒸馏损失 * alpha + 原始损失 * (1 - alpha)
        total_loss = config["distillation"]["alpha"] * total_loss + (1 - config["distillation"]["alpha"]) * original_loss

        if (self.state.global_step) % 100 == 0:
            self.log({
                "train/original_loss": original_loss.detach().cpu().item(),
                "train/logits_loss": logits_loss.detach().cpu().item(),
                "train/hidden_loss": avg_hidden_loss.detach().cpu().item(),
            })


        return total_loss

# Training arguments
training_arguments = TrainingArguments(**config["training"])

# Create the custom SFT Trainer
trainer = ComplexTrainer(
    model=student_model,
    teacher_model=teacher_model,
    adaptation_layer=adaptation_layer,  
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    #tokenizer=student_tokenizer,
    args=training_arguments,
    #max_seq_length=config["tokenizer"]["max_length"],
    #dataset_text_field="text",
)


# 添加动态冻结回调（仅当spectrum配置存在时）
if "spectrum" in config and "layers_to_unfreeze" in config["spectrum"]:
    freeze_callback = FreezeCallback(
        student_model=student_model,
        unfrozen_layers_file=config["spectrum"]["layers_to_unfreeze"]
    )
    trainer.add_callback(freeze_callback)
else:
    print("Spectrum configuration not found. All layers will remain trainable.")

# 自定义回调函数，用于在评估后记录更多信息
class EvalCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            print(f"\nEvaluation results: {metrics}")


# 添加回调函数
trainer.add_callback(EvalCallback())

# Add the teacher model to the trainer
trainer.teacher_model = teacher_model

# Train the model
trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

# Save the final model
trainer.save_model(config["training"]["output_dir"])