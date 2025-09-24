from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from jinja2 import Template

# --------------------------
# 1️⃣ 加载模型和 tokenizer
# --------------------------
model_name = "./distilled_qwen3_instruct"  # 替换成你本地或远程模型路径
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
if torch.cuda.is_available():
    model.cuda()

# --------------------------
# 2️⃣ 定义 ChatML 模板
# --------------------------
template_str = """
{% for message in messages %}
{% if loop.first and messages[0]['role'] != 'system' %}
<|im_start|>system
You are a helpful assistant.<|im_end|>
{% endif %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}
<|im_start|>assistant
"""

template = Template(template_str)

# --------------------------
# 3️⃣ 定义对话函数
# --------------------------
def chat(messages, max_new_tokens=1024, temperature=0.7):
    """
    messages: List[Dict], 每条 dict 形如 {"role": "user"/"assistant", "content": "内容"}
    """
    prompt = template.render(messages=messages)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    reply = tokenizer.decode(generated, skip_special_tokens=True)
    return reply

# --------------------------
# 4️⃣ 循环对话
# --------------------------
if __name__ == "__main__":
    print("开始和模型对话（输入 'exit' 或 'quit' 退出）")
    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("对话结束")
            break
        
        # 添加用户消息到历史
        history.append({"role": "user", "content": user_input})
        
        # 调用模型生成回复
        response = chat(history)
        print("Assistant:", response)
        
        # 将模型回复加入历史
        history.append({"role": "assistant", "content": response})
