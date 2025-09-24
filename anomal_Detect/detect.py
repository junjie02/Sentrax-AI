import pickle
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import json


# 指定需要提取的字段列表
TARGET_FIELDS = [
    'repeat_count', 'confidence', 'vuln_name', 'attack_type', 'type', 'is_web_attack',
    'super_type', 'vuln_desc', 'detail_info', 'dport','dip_addr', 'h_method', 
    'h_proto_version', 'uri', 'h_url', 'url_path', 'user-agent', 
    'accept', 'req_header', 'req_body', 'rule_key', 'rsp_body',  
    'rule_desc', 'rule_labels', 'hazard_level', 'hazard_rating', 'vuln_harm',
    'rsp_header', 'alarm_id'
]

class SimpleLogDetector:
    def __init__(self, model_path: str, llm_model_path: str = "./Qwen3-Embedding-0.6B"):
        """
        简单的日志异常检测器
        
        Args:
            model_path: 训练好的.pkl模型文件路径
            llm_model_path: LLM模型路径
        """
        # 加载LLM模型
        print("加载LLM模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(llm_model_path, trust_remote_code=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # 加载训练好的检测器
        print("加载检测器...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.detector = model_data['detector']
        self.scaler = model_data['scaler']
        self.detector_type = model_data['detector_type']
        self.threshold = model_data['threshold']
        self.knn_threshold = model_data.get('knn_threshold')
        
        print("✓ 模型加载完成！")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """获取文本的embedding"""
        text = " ".join(text.split())  # 简单预处理
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                               padding=True, max_length=8192).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            if hasattr(outputs, "pooler_output"):
                embeddings = outputs.pooler_output.cpu().numpy()
            else:
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        return embeddings.squeeze()
    
    def detect(self, log_text: str) -> dict:
        """
        检测单条日志
        
        Args:
            log_text: 日志文本
            
        Returns:
            检测结果字典
        """
        # 获取embedding
        embedding = self.get_embedding(log_text)
        embedding_scaled = self.scaler.transform([embedding])
        
        # 预测
        if self.detector_type == "isolation_forest":
            prediction = self.detector.predict(embedding_scaled)[0]
            score = self.detector.decision_function(embedding_scaled)[0]
            is_anomaly = (prediction == -1)
        elif self.detector_type == "knn":
            distances, _ = self.detector.kneighbors(embedding_scaled)
            score = distances.mean()
            is_anomaly = (score > self.knn_threshold)
        
        result = {
            'log': log_text,
            'is_anomaly': bool(is_anomaly),
            'score': float(score),
            'status': '异常' if is_anomaly else '正常'
        }
        
        return result
def load_dataset(text: str):
        
    text = json.loads(text)
    # 提取指定字段并构建键值对字符串
    field_pairs = []
    for field in TARGET_FIELDS:
        # 获取字段值，如果不存在则使用空字符串
        value = text.get(field, "无")
        # 确保值是字符串类型
        if not isinstance(value, str):
            value = str(value)
        # 添加键值对
        value = value.replace('\n', ' ').replace('\r', ' ').strip()  # 清理换行符
        field_pairs.append(f"{field}:{value}")
                
        # 拼接所有字段为一条日志
    log_text = "网络安全日志："+",".join(field_pairs)
    return log_text


# 使用示例
def main():
    # 初始化检测器
    detector = SimpleLogDetector(
        model_path="./saved_models/security_log_detector.pkl",  # 你的模型路径
        llm_model_path="./Qwen3-Embedding-0.6B"  # 你的LLM模型路径
    )
    
    # 检测多条日志
    test_logs = [
        "{\"write_date\": \"1732636848\", \"vlan_id\": \"\", \"vuln_type\": \"发现使用远程连接工具向日葵\", \"attack_type\": \"代理工具\", \"dip_group\": \"省公司\", \"is_web_attack\": \"0\", \"nid\": \"\", \"dip\": \"47.96.156.45\", \"sip_group\": \"\", \"repeat_count\": \"1\", \"type\": \"代理工具\", \"skyeye_type\": \"webids-ids_dolog\", \"_origin\": {\"write_date\": 1732636848, \"rule_name\": \"发现使用远程连接工具向日葵\", \"hit_field\": \"\", \"description\": \"1\", \"dip\": \"47.96.156.45\", \"protocol_id\": 6, \"hit_end\": 193, \"uri\": \"\", \"cnnvd_id\": \"\", \"dport\": 443, \"rule_version\": \"3.0.1122.14572\", \"hit_start\": 173, \"detail_info\": \"发现有主机在使用远程连接工具向日葵，向日葵是一款优质的远程支持、远程访问和在线协作软件。\", \"packet_size\": 264, \"appid\": 19, \"proto\": \"ssl\", \"xff\": \"\", \"sip\": \"172.17.17.43\", \"attack_method\": \"远程\", \"affected_system\": \"\", \"sig_id\": 2579, \"sport\": 17547, \"bulletin\": \"向日葵这类远程工具的使用是危险的，可能成为黑客的后门，请严格按照公司规定使用。\"}, \"hit_start\": \"173\", \"skyeye_id\": \"\", \"payload\": {\"packet_data\": \"ACSsIoQzACSs3Z37CABFAAD6TPlAADsGaTusERErL2CcLUSLAbtcH9BjDhHR41AYBAHcYQAAFgMBAM0BAADJAwPml6wUhXYY/a2x7aIMsvAdKfoIdi0j/srFlAqUvYDIxQAAOMAswDAAn8ypzKjMqsArwC8AnsAkwCgAa8AjwCcAZ8AKwBQAOcAJwBMAMwCdAJwAPQA8ADUALwD/AQAAaAAAAB4AHAAAGWFwaS1zdGQuc3VubG9naW4ub3JheS5jb20ACwAEAwABAgAKAAoACAAdABcAGQAYACMAAAANACAAHgYBBgIGAwUBBQIFAwQBBAIEAwMBAwIDAwIBAgICAwAWAAAAFwAA\"}, \"detail_info\": \"发现有主机在使用远程连接工具向日葵，向日葵是一款优质的远程支持、远程访问和在线协作软件。\", \"file_md5\": \"\", \"host\": \"\", \"host_state\": \"企图\", \"rule_key\": \"webids\", \"api\": \"\", \"first_access_time\": \"2024-11-27 00:00:48\", \"hazard_level\": \"4\", \"hazard_rating\": \"中危\", \"rule_name\": \"发现使用远程连接工具向日葵\", \"packet_data\": \"ACSsIoQzACSs3Z37CABFAAD6TPlAADsGaTusERErL2CcLUSLAbtcH9BjDhHR41AYBAHcYQAAFgMBAM0BAADJAwPml6wUhXYY/a2x7aIMsvAdKfoIdi0j/srFlAqUvYDIxQAAOMAswDAAn8ypzKjMqsArwC8AnsAkwCgAa8AjwCcAZ8AKwBQAOcAJwBMAMwCdAJwAPQA8ADUALwD/AQAAaAAAAB4AHAAAGWFwaS1zdGQuc3VubG9naW4ub3JheS5jb20ACwAEAwABAgAKAAoACAAdABcAGQAYACMAAAANACAAHgYBBgIGAwUBBQIFAwQBBAIEAwMBAwIDAwIBAgICAwAWAAAAFwAA\", \"hit_field\": \"\", \"sip_addr\": \"中国--湖南省--长沙市\", \"protocol_id\": \"6\", \"cnnvd_id\": \"\", \"x_forwarded_for\": \"\", \"device_ip\": \"172.31.191.7\", \"alarm_source\": \"天眼分析平台-8\", \"alarm_sip\": \"172.17.17.43\", \"rule_desc\": \"网络攻击\", \"rule_version\": \"3.0.1122.14572\", \"skyeye_index\": \"\", \"sip_ioc_dip\": \"928bfe2f28ea05d4c3f222997cfe925d\", \"proto\": \"ssl\", \"xff\": \"\", \"alarm_id\": \"20241127_7b49d204432a5ebd3e7dc7f6691a8e9e\", \"attack_chain\": \"0x02020000\", \"access_time\": \"2024-11-27 00:00:48\", \"attack_addr\": \"\", \"type_chain\": \"16220000\", \"description\": \"1\", \"dip_addr\": \"中国--浙江省--杭州市\", \"dport\": \"443\", \"alert_devip\": \"172.31.191.8\", \"rsp_status\": \"\", \"update_time\": \"1732636791\", \"branch_id\": \"QbJK/fzEi\", \"att_ck\": \"初始访问:TA0001|利用面向公众的应用程序:T1190\", \"sip\": \"172.17.17.43\", \"attack_method\": \"远程\", \"affected_system\": \"\", \"dimension\": \"3\", \"skyeye_serial_num\": \"QbJK/jYtc\", \"src_mac\": \"00:24:ac:dd:9d:fb\", \"confidence\": \"高\", \"super_type\": \"攻击利用\", \"super_attack_chain\": \"0x02000000\", \"hit_end\": \"193\", \"serial_num\": \"QbJK/jYtc\", \"uri\": \"\", \"dst_mac\": \"00:24:ac:22:84:33\", \"is_delete\": \"0\", \"tcp_option\": \"\", \"rule_id\": \"0x5a11\", \"attack_org\": \"\", \"is_white\": \"0\", \"packet_size\": \"264\", \"alarm_sample\": \"1\", \"appid\": \"19\", \"attack_sip\": \"\", \"rule_state\": \"green\", \"asset_group\": \"省公司\", \"ioc\": \"23057-发现使用远程连接工具向日葵\", \"rule_labels\": \"{\\\"dns\\\": {\\\"dns\\\": [{\\\"request\\\": \\\"\\\", \\\"answers\\\": [{\\\"type\\\": \\\"A\\\", \\\"data\\\": \\\"T1190\\\"}]}]}}\", \"sig_id\": \"2579\", \"host_md5\": \"\", \"sport\": \"17547\", \"bulletin\": \"向日葵这类远程工具的使用是危险的，可能成为黑客的后门，请严格按照公司规定使用。\"}",
    ]
    
    print("\n" + "="*50)
    for log in test_logs:
        log = load_dataset(log)
        result = detector.detect(log)
        print(f"{result['status']:>4} | {result['score']:>8.4f} | {result['log']}")


if __name__ == "__main__":
    main()