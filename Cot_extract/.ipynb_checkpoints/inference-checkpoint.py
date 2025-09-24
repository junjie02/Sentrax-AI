import json
import os
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams

class SecurityLogAnalyzer:
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        初始化 vLLM 模型
        """
        self.device = device
        print("加载 vLLM 模型中...")
        
        # 优化vLLM参数设置
        self.llm = LLM(
            model=model_path, 
            tensor_parallel_size=1, 
            dtype="bfloat16" if device == "cuda" else "float16",
            max_model_len=2560,  # 设置合理的最大长度
            trust_remote_code=True  # 对于某些模型可能需要
        )
        print(f"模型已加载到设备: {self.device}")

        # 优化采样参数
        self.sampling_params = SamplingParams(
            max_tokens=1024,  # 明确限制生成的新token数量最多为300
            temperature=0.5,  # 降低温度，提高一致性
            top_p=0.7,
            repetition_penalty=1.2,
            stop=["分析结束", "结束", "</s>", "<|endoftext|>", "<|im_end|>"]  # 添加停止符
        )

    def create_analysis_prompt(self, log_input: str, log_output: str) -> str:
        """
        构建分析日志的prompt
        """
        prompt_template = """你是一个网络安全研判专家模型，请根据网络安全告警日志和判定结果给出分析解释。网络安全日志：{log_content}判定结果：{judgment}请分析为什么判定为"{judgment}"：给出一段分析即可，不需要有其他无关内容，限制100字以内。语言尽量简介，回答时不要根据结论分析，而是要用根据分析过程推断出结论的语气。"""

        return prompt_template.format(
            log_content=log_input,
            judgment=log_output
        )

    def clean_generated_text(self, text: str) -> str:
        """
        清理生成的文本
        """
        # 移除思考标签和内容
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 移除元话语和说明性文本
        unwanted_patterns = [
            r'此答案.*?(?=\n|$)', 
            r'仅作为示例.*?(?=\n|$)',
            r'未经许可.*?(?=\n|$)',
            r'所有权利.*?(?=\n|$)',
            r'END OF.*?(?=\n|$)',
            r'```.*?```',
            r'---.*',
            r'关键词提示.*',
            r'值得注意的是.*?(?=\n|$)',
            r'注意.*?(?=\n|$)',
            r'请注意.*?(?=\n|$)',
        ]
        
        for pattern in unwanted_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
            
        # 移除特殊标记
        special_tokens = ['<|im_start|>', '<|im_end|>', '<|endoftext|>', '</s>']
        for token in special_tokens:
            text = text.replace(token, '')
        
        # 移除多余的换行和空格
        text = re.sub(r'\n\s*\n', '\n', text)  # 多个换行合并为一个
        text = text.strip()
        
        # 如果文本以常见的前缀开始，移除它们
        prefixes = ['分析：', '安全分析：', '分析解释：', '解释：']
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        
        return text

    def generate_analysis_batch(self, prompts: list) -> list:
        """
        批量生成分析
        """
        outputs = self.llm.generate(prompts, sampling_params=self.sampling_params)
        
        analyses = []
        for output in outputs:
            raw_text = output.outputs[0].text
            cleaned_text = self.clean_generated_text(raw_text)
            analyses.append(cleaned_text)
        
        return analyses

    def process_dataset(self, input_file: str, output_file: str, num_samples: int = 100):
        """
        批量处理数据集并生成分析
        """
        print(f"读取数据文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 限制处理数量
        if len(data) > num_samples:
            data = data[:num_samples]

        print(f"将处理 {len(data)} 条样本")

        # 准备有效数据
        valid_data = []
        prompts = []
        
        for i, item in enumerate(data):
            original_input = item.get("input", "")
            original_output = item.get("output", "")

            if not original_input or not original_output:
                print(f"第 {i+1} 条数据缺少必要字段，跳过")
                continue

            valid_data.append(item)
            prompts.append(self.create_analysis_prompt(original_input, original_output))

        if not prompts:
            print("没有有效数据可处理")
            return

        print(f"开始批量生成分析...")
        
        # 批量生成分析
        analyses = self.generate_analysis_batch(prompts)

        # 构建最终数据
        enhanced_data = []
        for item, analysis in zip(valid_data, analyses):
            enhanced_item = {
                "input": item["input"],
                "output": f"<reasoning>{analysis}</reasoning>\n\n<answer>{item['output']}</answer>"
            }
            enhanced_data.append(enhanced_item)

        # 保存结果
        print(f"保存结果到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, ensure_ascii=False, indent=2)

        print(f"完成！成功处理 {len(enhanced_data)} 条数据")

        # 显示示例
        if enhanced_data:
            print("\n=== 处理示例 ===")
            sample = enhanced_data[0]
            print("原始判定:", sample['output'].split('\n\n分析解释：')[0])
            analysis_part = sample['output'].split('\n\n分析解释：')[1] if '\n\n分析解释：' in sample['output'] else ""
            print(f"生成分析: {analysis_part[:300]}...")

def main():
    """
    主函数
    """
    # 配置参数
    MODEL_PATH = "./secGPT"
    INPUT_FILE = "train_dataset_oversampled.json"
    OUTPUT_FILE = "train_dataset_cot2.json"
    NUM_SAMPLES = 100#53280  # 改回合理数量
    
    # 检查输入文件
    if not os.path.exists(INPUT_FILE):
        print(f"错误：输入文件 {INPUT_FILE} 不存在！")
        return

    try:
        # 创建分析器并处理
        analyzer = SecurityLogAnalyzer(MODEL_PATH)
        analyzer.process_dataset(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
            num_samples=NUM_SAMPLES
        )
        
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()