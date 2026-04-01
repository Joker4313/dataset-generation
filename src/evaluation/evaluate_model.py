import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import classification_report

def main():
    parser = argparse.ArgumentParser(description="意图分类大模型验证集效果评估打分脚本")
    parser.add_argument("--model_path", type=str, required=True, help="使用 LLaMA-Factory 导出的合并模型路径，或在线模型库路径如 Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--test_file", type=str, default="data/val.jsonl", help="验证集文件路径")
    parser.add_argument("--max_samples", type=int, default=-1, help="最多评估样本数量（便于测试环境快速出表，-1 为全部）")
    args = parser.parse_args()

    print(f"[*] 加载基座与分词器： {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16,  # 兼容所有显卡，若您的卡较新可以修改为 bfloat16
        trust_remote_code=True
    ).eval()
    
    print(f"[*] 读取验证集： {args.test_file} ...")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    if args.max_samples > 0:
        lines = lines[:args.max_samples]
        
    y_true = []
    y_pred = []
    
    print("[*] 开始批量推理验证 (为保证分类置信度，设置贪心采样的 temperature=0.01)...")
    for line in tqdm(lines, desc="Evaluating"):
        record = json.loads(line)
        messages = record.get("messages", [])
        
        # 确保数据格式完备（至少包含 System, User, Assistant 三句话）
        if len(messages) < 3:
            continue
            
        # 巧妙切分问题与答案
        context_messages = messages[:2]       # [System, User]
        true_label = messages[2]["content"].strip() # Assistant (Truth Label)
        
        # 应用 Qwen 内置的 ChatML 对话格式模板
        text = tokenizer.apply_chat_template(context_messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=16, # 我们只做意图分类名词，最长不过十几个字，节省时间
                temperature=0.01,
                do_sample=False
            )
            
        # 从输出序列中减去输入序列（Prompt）长度，剥离出纯净化答案
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        pred_label = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        
    print("\n\n" + "="*20 + " 意图分类评估效能报告 " + "="*20 + "\n")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
