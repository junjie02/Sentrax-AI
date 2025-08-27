# data_utils.py
from datasets import DatasetDict, load_dataset, Features, Sequence, Value
from transformers import PreTrainedTokenizer
import torch

def load_and_shuffle_dataset(config: dict) -> DatasetDict:
    """
    加载数据集并打乱（按config配置）
    Args:
        config: 总配置字典（需包含dataset相关字段）
    Returns:
        打乱后的数据集（未预处理）
    """
    dataset = load_dataset(
        config["dataset"]["name"],
        split=config["dataset"]["split"]
    )
    dataset = dataset.shuffle(seed=config["dataset"]["seed"])
    
    if "num_samples" in config["dataset"] and config["dataset"]["num_samples"] > 0:
        dataset = dataset.select(range(config["dataset"]["num_samples"]))
    
    return dataset


def prepare_dataset(example: dict, student_tokenizer: PreTrainedTokenizer, config: dict) -> dict:
    """
    单样本预处理：将ShareGPT格式转换为模型输入格式（适配学生/教师tokenizer）
    Args:
        example: 单条数据集样本（需含"conversations"字段）
        student_tokenizer: 学生模型的tokenizer
        config: 总配置字典（需包含tokenizer相关字段）
    Returns:
        预处理后的样本（含input_ids、attention_mask等）
    """
    conversations = example.get("conversations", [])
    message = [] # 包含gpt
    # 解析ShareGPT格式的对话
    if isinstance(conversations, list):
        for conv in conversations:
            if not isinstance(conv, dict):
                continue
            conv_from = conv.get("from")
            conv_value = conv.get("value", "").strip()
            if not conv_value:
                continue
            
            if conv_from == "human":
                message.append({"role": "user", "content": conv_value})
            elif conv_from == "gpt":
                message.append({"role": "assistant", "content": conv_value})
            elif conv_from == "system":
                message.insert(0, {"role": "system", "content": conv_value})
    
    # 补充默认 system prompt
    if not any(msg.get("role") == "system" for msg in message):
        message.insert(0, {"role": "system", "content": "You are a helpful assistant."})

    # 应用 chat template 生成文本
    student_text = student_tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=False
    )

    max_len = config["tokenizer"]["max_length"]
    student_enc = student_tokenizer(
        student_text, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt"
    )
    # 生成 labels
    labels = student_enc["input_ids"].clone()
    
    # 设置非 assistant token 为 -100
    non_assistant_message = [msg for msg in message if msg["role"] != "assistant"]
    non_assistant_text = student_tokenizer.apply_chat_template(
        non_assistant_message, tokenize=False, add_generation_prompt=True
    )
    non_assistant_enc = student_tokenizer(
        non_assistant_text, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt"
    )
    non_assistant_len = non_assistant_enc["attention_mask"].sum().item()
    labels[:, :non_assistant_len] = -100
    #查看labels内容：
    #decoded_labels = student_tokenizer.decode(
    #    [int(x) for x in labels[0].tolist() if x != -100],
    #    skip_special_tokens=False
    #) 
    #print("Decoded labels:", decoded_labels)
    result = {
        "input_ids": student_enc["input_ids"].squeeze(0).tolist(),
        "attention_mask": student_enc["attention_mask"].squeeze(0).tolist(),
        "labels": labels.squeeze(0).tolist()
    }

    return result


def preprocess_and_split_dataset(dataset, student_tokenizer, config: dict) -> DatasetDict:
    """
    批量预处理数据集 + 划分训练/测试集
    Args:
        dataset: 原始数据集（未预处理）
        student_tokenizer: 学生模型的tokenizer
        config: 总配置字典
    Returns:
        划分后的数据集（含train/test，已预处理）
    """
    print("Preprocessing and tokenizing dataset...")
    original_columns = dataset.column_names
    num_proc = config.get("dataset", {}).get("num_proc", 8)

    features = Features({
        "input_ids": Sequence(Value("int64")),
        "attention_mask": Sequence(Value("int64")),
        "labels": Sequence(Value("int64")),
    })

    processed_dataset = dataset.map(
        lambda example: prepare_dataset(example, student_tokenizer, config),
        remove_columns=original_columns,
        num_proc=num_proc,
        desc="Tokenizing dataset",
        features=features,
    )

    test_size = config.get("dataset", {}).get("test_size", 0.1)
    split_dataset = processed_dataset.train_test_split(
        test_size=test_size,
        seed=config["dataset"]["seed"]
    )

    print("Dataset preparation complete.")
    print("原始字段: ", original_columns)
    print("预处理后字段: ", processed_dataset.column_names)

    return split_dataset
