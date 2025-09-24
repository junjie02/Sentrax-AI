from dataset_pre import load_jsonl_dataset

DATASET_PATH = r"./cybersecurity_alarm_analysis_508/train_external_alpaca.jsonl"  # JSONL数据集路径

logs, labels = load_jsonl_dataset(DATASET_PATH)

print(logs[10])