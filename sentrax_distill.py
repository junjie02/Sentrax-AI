import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback
import yaml
from accelerate import Accelerator
from dataset.data_utils import load_and_shuffle_dataset, preprocess_and_split_dataset
from components.spectrum import FreezeCallback
from components.hidden_reflect import MultiLayerAdaptationLayer
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


# 设置wandb 
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_PROJECT'] = config["project_name"]
os.environ["WANDB_DIR"] = "./wandb_logs"

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
    student_tokenizer=student_tokenizer,
    teacher_tokenizer=teacher_tokenizer,
    config=config
)

# 使用flash_attention2加速
model_kwargs = {"torch_dtype": torch.bfloat16}
if config["model_config"]["use_flash_attention"]:
    model_kwargs["attn_implementation"] = "flash_attention_2"

# 加载模型
teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"], **model_kwargs).to(accelerator.device)
student_model = AutoModelForCausalLM.from_pretrained(config["models"]["student"], **model_kwargs).to(accelerator.device)

# 创建适配层
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

# 训练参数
training_arguments = TrainingArguments(**config["training"])

# Create the custom SFT Trainer
trainer = ComplexTrainer(
    model=student_model,
    teacher_model=teacher_model,
    adaptation_layer=adaptation_layer,  
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_arguments,
)

# 添加动态冻结回调（仅当spectrum配置存在时）
if "spectrum" in config and "layers_to_unfreeze" in config["spectrum"]:
    freeze_callback = FreezeCallback(
        student_model=student_model,
        unfrozen_layers_file=config["spectrum"]["layers_to_unfreeze"]
    )
    trainer.add_callback(freeze_callback)
else:
    print("No Spectrum. All layers will be trainable.")


# 添加教师模型到trainer
trainer.teacher_model = teacher_model

# 训练模型
trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

# 保存模型
trainer.save_model(config["training"]["output_dir"])