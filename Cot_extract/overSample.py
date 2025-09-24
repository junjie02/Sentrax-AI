import json
import random

def build_balanced_dataset(input_file: str, output_file: str, attack_times: int = 8, false_alarm_size: int = 25000):
    """
    构建新的数据集：
    - 攻击样本复制 attack_times 倍
    - 从误报样本中抽取 false_alarm_size 条
    - 合并并打乱，生成新数据集
    """
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except Exception as e:
                print(f"跳过格式错误的行: {line[:50]}... 错误: {e}")

    print(f"原始样本总数: {len(data)}")

    # 分类
    attack_samples = [item for item in data if item.get("output") == "攻击"]
    false_alarm_samples = [item for item in data if item.get("output") == "误报"]

    print(f"攻击样本数量: {len(attack_samples)}")
    print(f"误报样本数量: {len(false_alarm_samples)}")

    if not attack_samples:
        print("没有找到攻击样本，退出")
        return
    if not false_alarm_samples:
        print("没有找到误报样本，退出")
        return

    # 攻击样本过采样
    attack_oversampled = attack_samples * attack_times
    print(f"过采样后攻击样本数: {len(attack_oversampled)}")

    # 抽取误报样本
    if len(false_alarm_samples) > false_alarm_size:
        false_alarm_selected = random.sample(false_alarm_samples, false_alarm_size)
    else:
        false_alarm_selected = false_alarm_samples
        print(f"误报不足 {false_alarm_size} 条，实际取 {len(false_alarm_selected)}")

    # 合并
    new_data = attack_oversampled + false_alarm_selected
    random.shuffle(new_data)

    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in new_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"新数据集已保存到 {output_file}，总样本数: {len(new_data)}")
    print(f"攻击样本占比: {len(attack_oversampled) / len(new_data):.2%}")
    print(f"误报样本占比: {len(false_alarm_selected) / len(new_data):.2%}")


if __name__ == "__main__":
    input_file = "train_dataset.jsonl"           # 原始数据集
    output_file = "train_balanced.jsonl"         # 新的数据集
    build_balanced_dataset(input_file, output_file, attack_times=80, false_alarm_size=30000)
