import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import yaml
from accelerate import Accelerator
from dataset.data_utils import load_and_shuffle_dataset, preprocess_and_split_dataset
from components.trainer import ComplexTrainer

#加载配置文件
CONFIG_PATH = "./config.yaml"
# 读取YAML文件并加载配置
def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)  # safe_load避免加载恶意代码
    return config

# 执行加载
config = load_config(CONFIG_PATH)

accelerator = Accelerator()
device = accelerator.device

# 加载分词器
teacher_tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"])
student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])
student_tokenizer.chat_template = config["tokenizer"]["chat_template"]

#加载数据集
raw_dataset = load_and_shuffle_dataset(config)

# 3. 预处理数据集 + 划分train/test
dataset = preprocess_and_split_dataset(
    dataset=raw_dataset,
    tokenizer=student_tokenizer,
    config=config
)

# 使用flash_attention2加速
model_kwargs = {"torch_dtype": torch.bfloat16}
if config["model_config"]["use_flash_attention"]:
    model_kwargs["attn_implementation"] = "flash_attention_2"

# 加载模型
teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"], **model_kwargs).to(accelerator.device)
student_model = AutoModelForCausalLM.from_pretrained(config["models"]["student"], **model_kwargs).to(accelerator.device)

# 训练参数
training_arguments = TrainingArguments(**config["training"])

# Create the custom SFT Trainer
trainer = ComplexTrainer(
    model=student_model,
    teacher_model=teacher_model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_arguments,
)

# 添加教师模型到trainer
trainer.teacher_model = teacher_model

# 训练模型
trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

# 保存模型
trainer.save_model(config["training"]["output_dir"])