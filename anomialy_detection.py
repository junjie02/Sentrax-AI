import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

class SecurityLogAnomalyDetector:
    """
    基于LLM Embedding + One-Class检测的网络安全日志异常检测器
    支持最近邻和Isolation Forest两种异常检测算法
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 detector_type: str = "isolation_forest", contamination: float = 0.1):
        """
        初始化异常检测器
        
        Args:
            model_name: 预训练模型名称，建议使用安全领域微调的模型
            detector_type: 检测器类型 ('isolation_forest' or 'knn')
            contamination: 异常比例估计
        """
        self.model_name = model_name
        self.detector_type = detector_type
        self.contamination = contamination
        
        # 初始化LLM模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 初始化检测器
        if detector_type == "isolation_forest":
            self.detector = IsolationForest(contamination=contamination, random_state=42)
        elif detector_type == "knn":
            self.detector = NearestNeighbors(n_neighbors=5, metric='cosine')
            self.knn_threshold = None
        else:
            raise ValueError("detector_type must be 'isolation_forest' or 'knn'")
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def extract_security_features(self, log_data: Dict[str, Any]) -> str:
        """
        从安全日志中提取关键特征，构造用于embedding的文本
        
        Args:
            log_data: 原始日志数据字典
            
        Returns:
            构造的特征文本
        """
        # 提取关键安全特征
        features = []
        
        # 攻击类型和漏洞信息
        if 'vuln_type' in log_data:
            features.append(f"漏洞类型: {log_data['vuln_type']}")
        if 'attack_type' in log_data:
            features.append(f"攻击类型: {log_data['attack_type']}")
        if 'hazard_rating' in log_data:
            features.append(f"危险等级: {log_data['hazard_rating']}")
            
        # 网络信息
        if 'sip' in log_data and 'dip' in log_data:
            features.append(f"源IP: {log_data['sip']} 目标IP: {log_data['dip']}")
        if 'dport' in log_data:
            features.append(f"目标端口: {log_data['dport']}")
        if 'proto' in log_data:
            features.append(f"协议: {log_data['proto']}")
            
        # 地理位置信息
        if 'sip_addr' in log_data:
            features.append(f"源地址: {log_data['sip_addr']}")
            
        # URL和请求信息
        if 'uri' in log_data and log_data['uri']:
            features.append(f"请求URI: {log_data['uri']}")
        if 'h_method' in log_data:
            features.append(f"HTTP方法: {log_data['h_method']}")
            
        # Payload信息（关键）
        if 'req_body' in log_data and log_data.get('req_body'):
            # 这里可以解码base64的payload进行分析
            features.append(f"请求体存在")
            
        # 规则和置信度信息
        if 'rule_name' in log_data:
            features.append(f"触发规则: {log_data['rule_name']}")
        if 'confidence' in log_data:
            features.append(f"置信度: {log_data['confidence']}")
            
        # 攻击链信息
        if 'att_ck' in log_data and log_data['att_ck']:
            features.append(f"ATT&CK: {log_data['att_ck']}")
            
        return " | ".join(features)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        使用LLM获取文本embedding
        
        Args:
            text: 输入文本
            
        Returns:
            文本的embedding向量
        """
        # 文本预处理，截断过长文本
        if len(text) > 512:
            text = text[:512]
            
        # 获取embedding
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                              padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS] token或平均池化
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
        return embeddings.squeeze()
    
    def preprocess_logs(self, logs: List[Dict[str, Any]]) -> np.ndarray:
        """
        预处理日志数据，提取特征并生成embeddings
        
        Args:
            logs: 日志数据列表
            
        Returns:
            embeddings矩阵
        """
        embeddings = []
        
        print(f"处理 {len(logs)} 条日志...")
        for i, log in enumerate(logs):
            if i % 100 == 0:
                print(f"已处理: {i}/{len(logs)}")
                
            # 提取安全特征文本
            feature_text = self.extract_security_features(log)
            
            # 获取embedding
            embedding = self.get_embedding(feature_text)
            embeddings.append(embedding)
            
        return np.array(embeddings)
    
    def fit(self, normal_logs: List[Dict[str, Any]]):
        """
        使用正常日志训练异常检测器
        
        Args:
            normal_logs: 正常日志数据列表
        """
        print("开始训练异常检测器...")
        
        # 获取正常日志的embeddings
        normal_embeddings = self.preprocess_logs(normal_logs)
        
        # 标准化
        normal_embeddings_scaled = self.scaler.fit_transform(normal_embeddings)
        
        # 训练检测器
        if self.detector_type == "isolation_forest":
            self.detector.fit(normal_embeddings_scaled)
        elif self.detector_type == "knn":
            self.detector.fit(normal_embeddings_scaled)
            # 计算阈值：使用训练数据的距离分布
            distances, _ = self.detector.kneighbors(normal_embeddings_scaled)
            mean_distances = distances.mean(axis=1)
            self.knn_threshold = np.percentile(mean_distances, 95)  # 95%分位数作为阈值
            
        self.is_fitted = True
        print("训练完成!")
    
    def predict(self, logs: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测日志是否异常
        
        Args:
            logs: 待检测的日志数据列表
            
        Returns:
            (predictions, scores): 预测结果和异常分数
        """
        if not self.is_fitted:
            raise ValueError("检测器尚未训练，请先调用fit()方法")
            
        # 获取embeddings
        embeddings = self.preprocess_logs(logs)
        embeddings_scaled = self.scaler.transform(embeddings)
        
        if self.detector_type == "isolation_forest":
            # Isolation Forest: -1表示异常，1表示正常
            predictions = self.detector.predict(embeddings_scaled)
            scores = self.detector.decision_function(embeddings_scaled)
            # 转换为0/1格式：0=正常，1=异常
            predictions = (predictions == -1).astype(int)
            
        elif self.detector_type == "knn":
            distances, _ = self.detector.kneighbors(embeddings_scaled)
            mean_distances = distances.mean(axis=1)
            predictions = (mean_distances > self.knn_threshold).astype(int)
            scores = mean_distances
            
        return predictions, scores
    
    def analyze_log(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析单条日志，返回详细的分析结果
        
        Args:
            log_data: 单条日志数据
            
        Returns:
            分析结果字典
        """
        predictions, scores = self.predict([log_data])
        
        is_anomaly = predictions[0] == 1
        confidence_score = abs(scores[0])
        
        # 提取关键特征用于解释
        feature_text = self.extract_security_features(log_data)
        
        # 风险评估
        risk_factors = []
        if log_data.get('hazard_rating') == '高危':
            risk_factors.append("高危漏洞")
        if 'shell' in log_data.get('vuln_type', '').lower():
            risk_factors.append("Shell命令执行")
        if log_data.get('sip_addr') and '印度' in log_data['sip_addr']:
            risk_factors.append("海外IP访问")
        if 'nmap' in log_data.get('rule_name', '').lower():
            risk_factors.append("网络扫描行为")
            
        result = {
            'is_anomaly': is_anomaly,
            'prediction': '真实攻击' if is_anomaly else '误报',
            'confidence_score': float(confidence_score),
            'risk_level': 'HIGH' if is_anomaly and confidence_score > 0.5 else 'MEDIUM' if is_anomaly else 'LOW',
            'extracted_features': feature_text,
            'risk_factors': risk_factors,
            'recommendation': '建议立即处置' if is_anomaly and confidence_score > 0.5 else '需要人工复核' if is_anomaly else '可能为误报'
        }
        
        return result

def load_huggingface_dataset(data_path: str) -> List[Dict[str, Any]]:
    """
    加载HuggingFace格式的数据集
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        解析后的日志数据列表
    """
    logs = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        conversations = item.get('conversations', [])
        for conv in conversations:
            if conv.get('from') == 'human' and '告警日志数据' in conv.get('value', ''):
                # 提取JSON格式的日志数据
                content = conv['value']
                json_start = content.find('{')
                if json_start != -1:
                    json_str = content[json_start:]
                    try:
                        log_data = json.loads(json_str)
                        logs.append(log_data)
                    except json.JSONDecodeError:
                        continue
                        
    return logs

def main_example():
    """
    主函数示例：演示如何使用异常检测器
    """
    print("=== 网络安全日志异常检测系统 ===")
    
    # 初始化检测器
    detector = SecurityLogAnomalyDetector(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # 可替换为安全领域模型
        detector_type="isolation_forest",
        contamination=0.15  # 预期异常比例15%
    )
    
    # 示例：模拟正常日志数据（实际使用时从文件加载）
    normal_logs = [
        {
            "vuln_type": "正常访问", "attack_type": "无", "hazard_rating": "低危",
            "sip": "192.168.1.10", "dip": "192.168.1.100", "dport": "80",
            "uri": "/index.html", "h_method": "GET", "sip_addr": "内网",
            "confidence": "低", "rule_name": "正常规则"
        }
        # 实际使用时需要更多正常样本
    ]
    
    # 示例：测试日志（包含异常）
    test_logs = [
        {
            "vuln_type": "Shell命令执行(机器学习)", "attack_type": "命令执行", 
            "hazard_rating": "高危", "sip": "117.209.87.214", "dip": "172.31.193.243",
            "dport": "80", "uri": "/boaform/admin/formLogin?username=ec8&psd=ec8",
            "h_method": "GET", "sip_addr": "印度--卡纳塔克邦--班加罗尔",
            "confidence": "高", "rule_name": "Shell命令执行检测",
            "req_body": "shell命令内容", "att_ck": "初始访问:TA0001"
        },
        {
            "vuln_type": "发现NMAP探测行为（SSL）", "attack_type": "信息泄露",
            "hazard_rating": "中危", "sip": "46.101.223.77", "dip": "172.23.47.67",
            "dport": "443", "proto": "ssl", "sip_addr": "德国--黑森州--美因河畔法兰克福",
            "confidence": "中", "rule_name": "发现NMAP探测行为（SSL）"
        }
    ]
    
    # 注意：实际使用需要足够的正常样本进行训练
    print("注意：此为演示代码，实际使用需要加载足够的正常日志样本进行训练")
    print("建议正常样本数量 > 1000条")
    
    # 如果有真实数据文件，可以这样加载：
    # logs = load_huggingface_dataset("your_dataset.json")
    # normal_logs = [log for log in logs if some_normal_condition(log)]
    
    try:
        # 训练检测器（需要足够的正常样本）
        if len(normal_logs) > 0:
            detector.fit(normal_logs)
            
            # 预测异常
            predictions, scores = detector.predict(test_logs)
            
            print("\n=== 检测结果 ===")
            for i, (log, pred, score) in enumerate(zip(test_logs, predictions, scores)):
                result = detector.analyze_log(log)
                print(f"\n日志 {i+1}:")
                print(f"漏洞类型: {log.get('vuln_type', 'N/A')}")
                print(f"预测结果: {result['prediction']}")
                print(f"风险等级: {result['risk_level']}")
                print(f"置信分数: {result['confidence_score']:.3f}")
                print(f"风险因素: {', '.join(result['risk_factors'])}")
                print(f"建议: {result['recommendation']}")
                
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        print("请确保有足够的训练数据并检查模型下载是否成功")

if __name__ == "__main__":
    main_example()