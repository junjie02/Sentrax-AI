import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import torch
from transformers import AutoTokenizer, AutoModel
import warnings
import random
import pickle
import os
from datetime import datetime
from tqdm import tqdm
from dataset_pre import load_jsonl_dataset, split_data_for_anomaly_detection

warnings.filterwarnings('ignore')

class SecurityLogAnomalyDetector:
    """
    改进版基于LLM Embedding + One-Class检测的网络安全日志异常检测器
    支持模型保存/加载、数据集划分和性能评估
    """
    
    def __init__(self, model_name: str = "./Qwen3-Embedding-0.6B", 
                 detector_type: str = "isolation_forest", contamination: float = 0.05,
                 model_save_dir: str = "./saved_models"):
        """
        初始化异常检测器
        
        Args:
            model_name: 预训练模型名称
            detector_type: 检测器类型 ('isolation_forest' or 'knn')
            contamination: 异常比例估计
            model_save_dir: 模型保存目录
        """
        self.model_name = model_name
        self.detector_type = detector_type
        self.contamination = contamination
        self.model_save_dir = model_save_dir
        
        # 创建保存目录
        os.makedirs(model_save_dir, exist_ok=True)
        
        # 初始化LLM模型
        print(f"加载预训练模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # 为Qwen模型添加pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        print(f"使用设备: {self.device}")
        print(f"模型词汇表大小: {len(self.tokenizer)}")
        
        # 初始化检测器
        if detector_type == "isolation_forest":
            self.detector = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
        elif detector_type == "knn":
            self.detector = NearestNeighbors(n_neighbors=5, metric='cosine', n_jobs=-1)
            self.knn_threshold = None
        else:
            raise ValueError("detector_type must be 'isolation_forest' or 'knn'")
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.threshold = None
        
        # 检查模型是否正常加载
        self._validate_model()
        
    def _validate_model(self):
        """
        验证模型是否正常加载和工作
        """
        try:
            print("验证模型功能...")
            test_text = "这是一个测试文本"
            test_embedding = self.get_embedding(test_text)
            print(f"模型验证成功！Embedding维度: {len(test_embedding)}")
            self.embedding_dim = len(test_embedding)
        except Exception as e:
            print(f"模型验证失败: {e}")
            raise RuntimeError(f"模型加载或验证失败: {e}")
        
    def preprocess_log_text(self, log_text: str) -> str:
        """
        预处理日志文本（可选的预处理步骤）
        
        Args:
            log_text: 原始日志字符串
            
        Returns:
            预处理后的日志字符串
        """
        # 去除多余空格和换行
        log_text = " ".join(log_text.split())
            
        return log_text
    
    def get_embedding(self, text: str) -> np.ndarray:
        text = self.preprocess_log_text(text)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                padding=True, max_length=8192).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # 官方推荐取 last_hidden_state 的均值池化 或者直接用 pooler_output
            if hasattr(outputs, "pooler_output"):
                embeddings = outputs.pooler_output.cpu().numpy()
            else:
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        return embeddings.squeeze()


    def preprocess_logs(self, logs: List[str]) -> np.ndarray:
        """
        预处理日志数据，生成embeddings
        
        Args:
            logs: 日志字符串列表
            
        Returns:
            embeddings矩阵
        """
        embeddings = []
        
        print(f"处理 {len(logs)} 条日志...")
        for i, log_text in tqdm(enumerate(logs),total=len(logs)):
                
            try:
                # 获取embedding
                embedding = self.get_embedding(log_text)
                embeddings.append(embedding)
            except Exception as e:
                print(f"处理第 {i} 条日志时出错: {e}")
                print(f"日志内容: {log_text[:100]}...")
                # 跳过有问题的日志，或者用零向量代替
                if len(embeddings) > 0:
                    # 使用前一个embedding的形状创建零向量
                    print("日志出现问题")
                    zero_embedding = np.zeros_like(embeddings[0])
                    embeddings.append(zero_embedding)
                else:
                    # 如果是第一个就出错，获取模型的embedding维度
                    try:
                        # 尝试用一个简单文本获取embedding维度
                        sample_embedding = self.get_embedding("test")
                        embedding_dim = len(sample_embedding)
                    except:
                        # 如果还是失败，使用Qwen模型的默认维度
                        embedding_dim = 896  # Qwen3-0.6B的hidden size
                    embeddings.append(np.zeros(embedding_dim))
        
        if len(embeddings) == 0:
            raise ValueError("所有日志都无法处理，请检查数据格式")
            
        return np.array(embeddings)
    
    def fit(self, normal_logs: List[str]):
        """
        使用正常日志训练异常检测器
        
        Args:
            normal_logs: 正常日志字符串列表
        """
        print("开始训练异常检测器...")
        
        # 获取正常日志的embeddings
        normal_embeddings = self.preprocess_logs(normal_logs)
        
        # 标准化
        normal_embeddings_scaled = self.scaler.fit_transform(normal_embeddings)

        # 训练检测器
        if self.detector_type == "isolation_forest":
            self.detector.fit(normal_embeddings_scaled)
            # 计算95分位数作为阈值
            scores = self.detector.score_samples(normal_embeddings_scaled)
            threshold = np.percentile(scores, 5)  # 5%分位数作为异常阈值
            print(f"Isolation Forest 训练完成，异常阈值: {threshold:.4f}")
        elif self.detector_type == "knn":
            self.detector.fit(normal_embeddings_scaled)
            # 计算阈值：使用训练数据的距离分布
            distances, _ = self.detector.kneighbors(normal_embeddings_scaled)
            mean_distances = distances.mean(axis=1)
            self.knn_threshold = np.percentile(mean_distances, 95)  # 95%分位数作为阈值
            threshold = self.knn_threshold
            print(f"KNN 训练完成，距离阈值: {threshold:.4f}")
            
        self.is_fitted = True
        self.threshold = threshold
        print("训练完成!")
        return threshold
    
    def predict(self, logs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测日志是否异常
        
        Args:
            logs: 待检测的日志字符串列表
            
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
    
    def save_model(self, model_name: str = None):
        """
        保存训练好的模型到本地
        
        Args:
            model_name: 模型文件名，如果为None则使用时间戳
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，无法保存")
        
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"anomaly_detector_{self.detector_type}_{timestamp}"
        
        model_path = os.path.join(self.model_save_dir, f"{model_name}.pkl")
        
        # 准备保存的数据
        model_data = {
            'detector': self.detector,
            'scaler': self.scaler,
            'detector_type': self.detector_type,
            'contamination': self.contamination,
            'threshold': self.threshold,
            'knn_threshold': self.knn_threshold if hasattr(self, 'knn_threshold') else None,
            'is_fitted': self.is_fitted,
            'model_name': self.model_name
        }
        
        # 保存模型
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"模型已保存到: {model_path}")
        return model_path
    
    def load_model(self, model_path: str):
        """
        从本地加载训练好的模型
        
        Args:
            model_path: 模型文件路径
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # 恢复模型状态
        self.detector = model_data['detector']
        self.scaler = model_data['scaler']
        self.detector_type = model_data['detector_type']
        self.contamination = model_data['contamination']
        self.threshold = model_data['threshold']
        self.knn_threshold = model_data.get('knn_threshold')
        self.is_fitted = model_data['is_fitted']
        
        print(f"模型已从 {model_path} 加载")
    
    def evaluate(self, test_logs: List[str], test_labels: List[int]) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            test_logs: 测试日志列表
            test_labels: 真实标签列表 (0=正常, 1=异常)
            
        Returns:
            评估结果字典
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，无法评估")
        
        # 预测
        predictions, scores = self.predict(test_logs)
        
        # 计算各种评估指标
        precision, recall, f1, support = precision_recall_fscore_support(
            test_labels, predictions, average='binary', pos_label=1
        )
        
        # 计算混淆矩阵
        cm = confusion_matrix(test_labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # 计算额外指标
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # 生成分类报告
        class_report = classification_report(test_labels, predictions, 
                                           target_names=['Normal', 'Anomaly'],
                                           output_dict=True)
        
        evaluation_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'confusion_matrix': cm,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'classification_report': class_report,
            'test_size': len(test_logs),
            'anomaly_count': sum(test_labels),
            'normal_count': len(test_labels) - sum(test_labels)
        }
        
        return evaluation_results
    
    def print_evaluation_report(self, eval_results: Dict[str, Any]):
        """
        打印评估报告
        
        Args:
            eval_results: 评估结果字典
        """
        print("\n" + "="*50)
        print("模型性能评估报告")
        print("="*50)
        
        print(f"测试集大小: {eval_results['test_size']}")
        print(f"正常日志数量: {eval_results['normal_count']}")
        print(f"异常日志数量: {eval_results['anomaly_count']}")
        
        print(f"\n核心指标:")
        print(f"准确率 (Accuracy): {eval_results['accuracy']:.4f}")
        print(f"精确率 (Precision): {eval_results['precision']:.4f}")
        print(f"召回率 (Recall): {eval_results['recall']:.4f}")
        print(f"F1分数 (F1-Score): {eval_results['f1_score']:.4f}")
        print(f"特异性 (Specificity): {eval_results['specificity']:.4f}")
        
        print(f"\n混淆矩阵:")
        cm = eval_results['confusion_matrix']
        print(f"真负例 (TN): {eval_results['true_negatives']}")
        print(f"假正例 (FP): {eval_results['false_positives']}")
        print(f"假负例 (FN): {eval_results['false_negatives']}")
        print(f"真正例 (TP): {eval_results['true_positives']}")
        
        print(f"\n详细分类报告:")
        report = eval_results['classification_report']
        print(f"正常日志 - 精确率: {report['Normal']['precision']:.4f}, "
              f"召回率: {report['Normal']['recall']:.4f}, "
              f"F1: {report['Normal']['f1-score']:.4f}")
        print(f"异常日志 - 精确率: {report['Anomaly']['precision']:.4f}, "
              f"召回率: {report['Anomaly']['recall']:.4f}, "
              f"F1: {report['Anomaly']['f1-score']:.4f}")
        
        print("="*50)


def load_local_model_safely(model_path: str):
    """
    安全加载本地模型的辅助函数
    
    Args:
        model_path: 本地模型路径
        
    Returns:
        (tokenizer, model): 加载成功的模型组件
    """
    import os
    
    # 检查路径格式和存在性
    if not os.path.exists(model_path):
        # 尝试几种常见的路径格式
        alternative_paths = [
            model_path,
            os.path.join("/root", model_path.lstrip("/")),
            os.path.join("/autodl-tmp", os.path.basename(model_path)),
            os.path.abspath(model_path)
        ]
        
        found_path = None
        for path in alternative_paths:
            if os.path.exists(path):
                found_path = path
                break
        
        if not found_path:
            raise FileNotFoundError(f"找不到模型路径: {model_path}\n尝试了以下路径:\n" + 
                                  "\n".join(f"  - {p}" for p in alternative_paths))
        model_path = found_path
    
    # 转换为绝对路径
    model_path = os.path.abspath(model_path)
    print(f"使用模型路径: {model_path}")
    
    # 检查必要文件
    required_files = ['config.json']
    optional_files = ['tokenizer.json', 'tokenizer_config.json', 'vocab.txt']
    
    print("\n检查模型文件:")
    for file in required_files + optional_files:
        file_path = os.path.join(model_path, file)
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        file_type = "必需" if file in required_files else "可选"
        print(f"  {status} {file} ({file_type})")
        
        if file in required_files and not exists:
            raise FileNotFoundError(f"缺少必需文件: {file}")
    
    # 尝试加载模型
    print(f"\n开始加载模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        print("✓ Tokenizer 加载成功")
        
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        print("✓ Model 加载成功")
        
        return tokenizer, model
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        
        # 提供详细的调试信息
        print(f"\n调试信息:")
        print(f"模型路径: {model_path}")
        print(f"路径是否存在: {os.path.exists(model_path)}")
        print(f"是否为目录: {os.path.isdir(model_path)}")
        
        if os.path.exists(model_path):
            print("目录内容:")
            try:
                files = os.listdir(model_path)
                for file in sorted(files)[:10]:  # 只显示前10个文件
                    print(f"  - {file}")
                if len(files) > 10:
                    print(f"  ... 还有 {len(files) - 10} 个文件")
            except Exception as list_error:
                print(f"  无法列出目录: {list_error}")
        
        raise e


def main_example_improved():
    """
    改进的主函数示例：使用新的数据划分方式
    """
    print("=== 网络安全日志异常检测系统（改进版数据划分）===")
    
    # 初始化检测器
    detector = SecurityLogAnomalyDetector(
        model_name="./Qwen3-Embedding-0.6B",
        detector_type="knn",
        contamination='auto', 
        model_save_dir="./saved_models"
    )
    
    # 尝试加载真实数据文件
    data_path = "./cybersecurity_alarm_analysis/train_alpaca.jsonl"
    
    try:
        print("尝试加载训练数据...")
        logs, labels = load_jsonl_dataset(data_path)
        
        if len(logs) > 0:
            print(f"成功加载 {len(logs)} 条日志")
            print(f"正常日志: {labels.count(0)} 条")
            print(f"异常日志: {labels.count(1)} 条")
            print(f"异常比例: {labels.count(1) / len(labels) * 100:.2f}%")
            
            # 限制数据量以便快速测试
            if len(logs) > 10000:
                print("数据量较大，随机选择10000条进行测试...")
                indices = random.sample(range(len(logs)), 10000)
                logs = [logs[i] for i in indices]
                labels = [labels[i] for i in indices]
            
            # 使用改进的数据划分方法
            train_logs, test_logs, train_labels, test_labels = split_data_for_anomaly_detection(
                logs, labels, test_size=0.2, random_state=65536
            )
            
            # 验证训练集只包含正常样本
            if train_labels.count(1) > 0:
                print(f"警告: 训练集中仍包含 {train_labels.count(1)} 个异常样本！")
            else:
                print("✓ 训练集验证通过：只包含正常样本")
            
            # 使用正常日志进行训练（无监督学习）
            print(f"\n使用 {len(train_logs)} 条正常日志进行训练...")
            
            # 训练检测器
            threshold = detector.fit(train_logs)
            
            # 保存模型
            model_path = detector.save_model("security_log_detector_improved")
            
            # 在测试集上评估
            print("\n开始评估模型性能...")
            eval_results = detector.evaluate(test_logs, test_labels)
            detector.print_evaluation_report(eval_results)
            
            # 额外分析：分别看正常和异常样本的检测情况
            print("\n=== 详细分析 ===")
            predictions, scores = detector.predict(test_logs)
            
            # 分析正常样本的检测情况
            normal_indices = [i for i, label in enumerate(test_labels) if label == 0]
            normal_predictions = [predictions[i] for i in normal_indices]
            normal_fp_rate = sum(normal_predictions) / len(normal_predictions) if normal_predictions else 0
            print(f"正常样本误报率: {normal_fp_rate * 100:.2f}% ({sum(normal_predictions)}/{len(normal_predictions)})")
            
            # 分析异常样本的检测情况
            anomaly_indices = [i for i, label in enumerate(test_labels) if label == 1]
            anomaly_predictions = [predictions[i] for i in anomaly_indices]
            anomaly_detection_rate = sum(anomaly_predictions) / len(anomaly_predictions) if anomaly_predictions else 0
            print(f"异常样本检出率: {anomaly_detection_rate * 100:.2f}% ({sum(anomaly_predictions)}/{len(anomaly_predictions)})")
                
        else:
            print("未能加载到有效的日志数据，请检查文件路径和格式")
            
    except FileNotFoundError:
        print(f"数据文件未找到: {data_path}")
            
    except Exception as e:
        print(f"运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_example_improved()