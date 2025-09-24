from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizer
import torch

def load_and_shuffle_dataset(config: dict) -> DatasetDict:
    """
    加载数据集并打乱（按config配置）
    """
    dataset = load_dataset(
        config["dataset"]["name"],
        split=config["dataset"]["split"]
    )
    dataset = dataset.shuffle(seed=config["dataset"]["seed"])
    
    if "num_samples" in config["dataset"] and config["dataset"]["num_samples"] > 0:
        dataset = dataset.select(range(config["dataset"]["num_samples"]))
    
    return dataset


def prepare_and_tokenize_dataset(example: dict, tokenizer: PreTrainedTokenizer, config: dict) -> dict:
    """
    处理 ShareGPT 格式对话并直接 tokenization
    """
    conversations = example.get("conversations", [])
    messages = []

    if isinstance(conversations, list):
        for conv in conversations:
            if not isinstance(conv, dict):
                continue
            conv_from = conv.get("from")
            conv_value = conv.get("value", "").strip()
            if not conv_value:
                continue
            
            if conv_from == "human":
                messages.append({"role": "user", "content": conv_value})
            elif conv_from == "gpt":
                messages.append({"role": "assistant", "content": conv_value})
            elif conv_from == "system":
                messages.insert(0, {"role": "system", "content": conv_value})

    # 如果没有 system 消息，默认加一个
    if not any(msg.get("role") == "system" for msg in messages):
        messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})

    # 生成文本
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # 直接 tokenizer 成 input_ids + attention_mask
    tokenized = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=config["tokenizer"]["max_length"]
    )

    return tokenized


def preprocess_and_split_dataset(dataset, tokenizer: PreTrainedTokenizer, config: dict) -> DatasetDict:
    """
    批量预处理并划分训练/测试集
    """
    print("Preprocessing and tokenizing dataset...")
    original_columns = dataset.column_names
    num_proc = config.get("dataset", {}).get("num_proc", 8)

    processed_dataset = dataset.map(
        lambda example: prepare_and_tokenize_dataset(example, tokenizer, config),
        remove_columns=original_columns,
        num_proc=num_proc,
        desc="Tokenizing dataset",
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
