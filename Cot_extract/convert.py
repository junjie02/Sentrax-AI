import os
import json
from tqdm import tqdm
from typing import Tuple, List

TARGET_FIELDS = [
    # 一、核心标识与降噪决策字段（去重、筛选核心）
    'alarm_id',               # 告警唯一ID（去重核心）
    'rule_id',                # 规则唯一ID（定位误报规则）
    'rule_version_str',       # 规则版本（区分规则迭代差异）
    'repeat_count',           # 重复次数（合并重复告警）
    'confidence',             # 置信度（过滤低置信误报）
    'hazard_level',           # 风险等级数值（筛选高风险）
    'hazard_rating',          # 风险等级描述（辅助风险筛选）
    
    # 二、攻击与漏洞核心信息字段（判断攻击真实性）
    'vuln_name',              # 漏洞名称（明确漏洞类型）
    'attack_type',            # 攻击类型（区分攻击手段）
    'super_type',             # 攻击大类（归类攻击场景）
    'is_web_attack',          # 是否Web攻击（场景匹配验证）
    'vuln_desc',              # 漏洞描述（补充漏洞细节）
    'detail_info',            # 攻击详情（验证攻击真实性）
    'vuln_harm',              # 漏洞危害（明确影响范围）
    'attack_chain',           # 攻击链（识别完整攻击链路）
    
    # 三、网络与流量定位字段（定位攻击源/目标）
    'sip',                    # 源IP（攻击发起方定位）
    'sip_addr',               # 源IP地址详情（归属地辅助判断）
    'dip',                    # 目的IP（攻击目标定位）
    'dip_addr',               # 目的IP地址详情（业务属性验证）
    'attack_addr',            # 攻击源地址（含经纬度，补充定位）
    'sip_ioc_dip',            # 源IP-IoC-目的IP关联（恶意指标匹配）
    'sport',                  # 源端口（异常端口识别）
    'dport',                  # 目的端口（服务场景匹配）
    'proto',                  # 网络协议（协议类型验证）
    
    # 四、设备与环境关联字段（过滤非核心设备误报）
    'device_ip',              # 检测设备IP（定位告警设备）
    'src_mac',                # 源MAC地址（物理设备定位）
    'dst_mac',                # 目的MAC地址（目标设备定位）
    
    # 五、HTTP/Web请求详情字段（Web场景合法性验证）
    'h_method',               # HTTP方法（攻击手段匹配）
    'h_proto_version',        # HTTP协议版本（协议支持验证）
    'uri',                    # 请求URI（资源路径定位）
    'host',                   # HTTP Host头（业务域名验证）
    'api',                    # 请求API接口（合法接口筛选）
    'user-agent',             # 用户代理（请求来源识别）
    'xff',                    # X-Forwarded-For头（真实客户端IP定位）
    
    # 六、日志内容与规则字段（关键验证信息）
    'req_header',             # HTTP请求头（提取关键验证字段）
    'req_body',               # HTTP请求体（验证攻击语句）
    'rule_key',               # 规则关键字（辅助定位规则）
    
    # 七、时间字段（时间范围筛选）
    'write_date',             # 日志写入时间（超期日志筛选）
    'update_time',            # 日志更新时间（避免重复处理更新告警）
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
                    # 清理换行符和多余空格，避免JSON格式错误
                    value = value.replace('\n', ' ').replace('\r', ' ').strip()
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


# -------------------------- 新增：写入JSONL文件函数 --------------------------
def save_to_jsonl(logs: List[str], labels: List[int], output_file_path: str) -> None:
    """
    将处理后的日志和标签存入新的JSONL文件，格式为 {"input": 日志内容, "output": "攻击"/"误报"}
    
    Args:
        logs: 格式化后的日志列表（与labels一一对应）
        labels: 标签列表（0=误报, 1=攻击）
        output_file_path: 输出JSONL文件的路径
    """
    # 校验日志和标签长度一致（避免数据错位）
    if len(logs) != len(labels):
        raise ValueError("日志列表与标签列表长度不匹配，无法写入")
    
    # 确保输出目录存在（若路径含子目录，自动创建）
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 逐行写入JSONL文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for log, label in tqdm(zip(logs, labels), total=len(logs), desc="写入JSONL文件"):
            # 转换标签为文字描述（0→"误报"，1→"攻击"）
            output_label = "攻击" if label == 1 else "误报"
            # 构建单条JSON数据
            json_data = {
                "input": log,       # input为logs中的一条样本内容
                "output": output_label  # output为对应的"攻击"/"误报"
            }
            # 写入JSON字符串（ensure_ascii=False保留中文，indent=None确保单行格式）
            f.write(json.dumps(json_data, ensure_ascii=False, indent=None) + '\n')
    
    print(f"\n数据已成功写入JSONL文件：{output_file_path}")
    print(f"写入样本总数：{len(logs)} 条")
    print(f"其中攻击样本：{sum(labels)} 条，误报样本：{len(labels)-sum(labels)} 条")


# -------------------------- 新增：主函数（加载+写入流程） --------------------------
def main(input_jsonl_path: str, output_jsonl_path: str) -> None:
    """
    主流程：加载原始JSONL→处理数据→写入新JSONL
    
    Args:
        input_jsonl_path: 原始JSONL数据集路径
        output_jsonl_path: 处理后输出的JSONL路径
    """
    try:
        # 1. 加载并处理原始数据
        print(f"开始加载原始数据集：{input_jsonl_path}")
        logs, labels = load_jsonl_dataset(input_jsonl_path)
        
        # 2. 将处理后的数据写入新JSONL
        if logs and labels:  # 确保有有效数据才写入
            print(f"\n开始写入处理后的数据到：{output_jsonl_path}")
            save_to_jsonl(logs, labels, output_jsonl_path)
        else:
            print("\n无有效数据可写入，跳过文件生成")
    
    except Exception as e:
        print(f"流程执行失败：{e}")


# -------------------------- 执行入口（需手动指定输入输出路径） --------------------------
if __name__ == "__main__":
    # ！！！请根据你的实际文件路径修改以下两个参数！！！
    INPUT_FILE = "train_alpaca.jsonl"    # 你的原始JSONL文件路径（如："./data/raw_logs.jsonl"）
    OUTPUT_FILE = "train_dataset.jsonl"  # 输出新JSONL文件的路径（如："./data/processed_logs.jsonl"）
    # 启动主流程
    main(INPUT_FILE, OUTPUT_FILE)