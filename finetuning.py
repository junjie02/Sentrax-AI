import os
import torch
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback
import yaml
from accelerate import Accelerator
from components.spectrum import FreezeCallback
from dataset.data_utils_fine import load_and_shuffle_dataset, preprocess_and_split_dataset
from peft import LoraConfig

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

# 设置wandb 
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_PROJECT'] = config["project_name"]
os.environ["WANDB_DIR"] = "./wandb/"

accelerator = Accelerator()
device = accelerator.device

model = AutoModelForCausalLM.from_pretrained(config["models"]["student"]).to(device)
# 打印前20个参数名（注意力层通常在模型的早期参数中）
#for name, _ in list(model.named_parameters())[:20]:
#    print(name)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])
tokenizer.chat_template = config["tokenizer"]["chat_template"]

#加载数据集
raw_dataset = load_and_shuffle_dataset(config)

# 3. 预处理数据集 + 划分train/test
dataset = preprocess_and_split_dataset(
    dataset=raw_dataset,
    student_tokenizer=tokenizer,     
    config=config
)

# 使用flash_attention2加速
model_kwargs = {"torch_dtype": torch.bfloat16}
if config["model_config"]["use_flash_attention"]:
    model_kwargs["attn_implementation"] = "flash_attention_2"


# 加载待微调模型
model = AutoModelForCausalLM.from_pretrained(
    config["models"]["student"],
    **model_kwargs
).to(accelerator.device)

# 训练参数
training_arguments = TrainingArguments(**config["training"])

#应用Lora
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Create the custom SFT Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,  # 结合PEFT配置启用LoRA
    args=training_arguments,
)


# Train the model
trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

# Save the final model
trainer.save_model(config["training"]["output_dir"])