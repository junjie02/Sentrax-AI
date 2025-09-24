import os
import json
from tqdm import tqdm
from typing import Tuple, List

# 指定需要提取的字段列表
TARGET_FIELDS = [
    'repeat_count', 'confidence', 'vuln_name', 'attack_type', 'type', 'is_web_attack',
    'super_type', 'vuln_desc', 'detail_info', 'dport','dip_addr', 'h_method', 
    'h_proto_version', 'uri', 'h_url', 'url_path', 'user-agent', 
    'accept', 'req_header', 'req_body', 'rule_key', 'rsp_body',  
    'rule_desc', 'rule_labels', 'hazard_level', 'hazard_rating', 'vuln_harm',
    'rsp_header', 'alarm_id'
]

def load_jsonl_dataset(file_path: str) -> Tuple[List[str], List[int]]:
    """
    加载JSONL格式的安全日志数据集，提取指定字段并格式化
    
    Args:
        file_path: JSONL文件路径
        
    Returns:
        logs: 格式化后的日志文本列表，每条日志为"字段名:值,字段名:值..."格式
        labels: 标签列表（0=误报, 1=攻击）
    """
    logs = []
    labels = []
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in tqdm(enumerate(f, 1)):
            try:
                # 解析JSONL行
                data = json.loads(line.strip())
                
                # 提取input字段并解析为JSON
                if 'input' in data:
                    input_str = data['input']
                    try:
                        # 解析input中的JSON数据
                        input_data = json.loads(input_str)
                    except json.JSONDecodeError:
                        print(f"第 {line_num} 行的input字段不是有效的JSON格式")
                        input_data = {}
                else:
                    print(f"第 {line_num} 行缺少 'input' 字段")
                    input_data = {}
                
                # 提取指定字段并构建键值对字符串
                field_pairs = []
                for field in TARGET_FIELDS:
                    # 获取字段值，如果不存在则使用空字符串
                    value = input_data.get(field, "无")
                    # 确保值是字符串类型
                    if not isinstance(value, str):
                        value = str(value)
                    # 添加键值对
                    value = value.replace('\n', ' ').replace('\r', ' ').strip()  # 清理换行符
                    field_pairs.append(f"{field}:{value}")
                
                # 拼接所有字段为一条日志
                log_text = "网络安全日志："+",".join(field_pairs)
                # 提取标签（"攻击"=1, "误报"=0）
                label = 1 if data.get('output', '').strip() == '攻击' else 0
                
                logs.append(log_text)
                labels.append(label)
                
            except Exception as e:
                print(f"解析第 {line_num} 行出错: {e}")
                continue
    
    # --------- 去重逻辑 ---------
    seen = set()
    unique_logs = []
    unique_labels = []

    for log, label in zip(logs, labels):
        if log not in seen:
            seen.add(log)
            unique_logs.append(log)
            unique_labels.append(label)

    logs, labels = unique_logs, unique_labels
    # -----------------------------
    print(f"数据集加载完成，共 {len(logs)} 条有效数据")
    print(f"攻击样本数: {sum(labels)}")
    print(f"误报样本数: {len(labels) - sum(labels)}")
    return logs, labels


def split_data_for_anomaly_detection(logs, labels, test_size=0.1, random_state=65536):
    """
    为异常检测任务划分数据集：
    - 训练集：只包含正常样本
    - 测试集：包含所有异常样本 + 一部分正常样本
    
    Args:
        logs: 日志列表
        labels: 标签列表 (0=正常, 1=异常)
        test_size: 测试集占总数据的比例
        random_state: 随机种子
    
    Returns:
        train_logs, test_logs, train_labels, test_labels
    """
    import random
    from sklearn.model_selection import train_test_split
    
    # 设置随机种子
    random.seed(random_state)
    
    # 分离正常样本和异常样本
    normal_indices = [i for i, label in enumerate(labels) if label == 0]
    anomaly_indices = [i for i, label in enumerate(labels) if label == 1]
    
    normal_logs = [logs[i] for i in normal_indices]
    normal_labels = [labels[i] for i in normal_indices]
    anomaly_logs = [logs[i] for i in anomaly_indices]
    anomaly_labels = [labels[i] for i in anomaly_indices]
    
    print(f"原始数据统计:")
    print(f"  正常样本: {len(normal_logs)} 条")
    print(f"  异常样本: {len(anomaly_logs)} 条")
    print(f"  总计: {len(logs)} 条")
    
    # 计算需要的测试集大小
    total_size = len(logs)
    target_test_size = int(total_size * test_size)
    
    # 所有异常样本都放入测试集
    test_anomaly_count = len(anomaly_logs)
    
    # 计算需要多少正常样本放入测试集
    test_normal_count = target_test_size - test_anomaly_count
    
    # 检查是否有足够的正常样本
    if test_normal_count > len(normal_logs):
        print(f"警告: 需要 {test_normal_count} 个正常样本作为测试集，但只有 {len(normal_logs)} 个正常样本")
        test_normal_count = len(normal_logs) // 2  # 至少保留一半正常样本用于训练
        print(f"调整为使用 {test_normal_count} 个正常样本作为测试集")
    
    if test_normal_count <= 0:
        raise ValueError("正常样本数量不足，无法按指定比例划分数据集")
    
    # 从正常样本中随机选择一部分作为测试集
    test_normal_indices = random.sample(range(len(normal_logs)), test_normal_count)
    train_normal_indices = [i for i in range(len(normal_logs)) if i not in test_normal_indices]
    
    # 构建训练集（只包含正常样本）
    train_logs = [normal_logs[i] for i in train_normal_indices]
    train_labels = [normal_labels[i] for i in train_normal_indices]  # 全部为0
    
    # 构建测试集（包含所有异常样本 + 部分正常样本）
    test_logs = ([normal_logs[i] for i in test_normal_indices] + 
                 anomaly_logs)
    test_labels = ([normal_labels[i] for i in test_normal_indices] + 
                   anomaly_labels)
    
    # 打乱测试集顺序
    test_indices = list(range(len(test_logs)))
    random.shuffle(test_indices)
    test_logs = [test_logs[i] for i in test_indices]
    test_labels = [test_labels[i] for i in test_indices]
    
    print(f"\n数据集划分结果:")
    print(f"训练集:")
    print(f"  总数: {len(train_logs)} 条")
    print(f"  正常样本: {train_labels.count(0)} 条")
    print(f"  异常样本: {train_labels.count(1)} 条")
    print(f"  异常比例: {train_labels.count(1) / len(train_labels) * 100:.2f}%")
    
    print(f"测试集:")
    print(f"  总数: {len(test_logs)} 条")
    print(f"  正常样本: {test_labels.count(0)} 条")
    print(f"  异常样本: {test_labels.count(1)} 条")
    print(f"  异常比例: {test_labels.count(1) / len(test_labels) * 100:.2f}%")
    
    print(f"\n整体统计:")
    print(f"实际测试集比例: {len(test_logs) / (len(train_logs) + len(test_logs)) * 100:.2f}%")
    print(f"目标测试集比例: {test_size * 100:.2f}%")
    
    return train_logs, test_logs, train_labels, test_labels