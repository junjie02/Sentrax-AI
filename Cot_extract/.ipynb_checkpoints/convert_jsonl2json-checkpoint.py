import json

# 输入和输出文件路径
input_path = "/root/autodl-tmp/train_balanced.jsonl"
output_path = "/root/autodl-tmp/train_dataset_oversampled.json"

data_list = []

# 逐行读取 JSONL
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:  # 跳过空行
            data_list.append(json.loads(line))

# 写入标准 JSON 文件
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data_list, f, ensure_ascii=False, indent=2)

print(f"成功将 {input_path} 转换为 {output_path}，总共 {len(data_list)} 条数据")
