import torch
import random
import time
import calculator_ast as calc_ast
import calculator_vocab
import calculator_model
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_latest_checkpoint(checkpoint_dir, model, device):
    """
    加载指定目录下最近的 checkpoint，并将模型参数恢复。
    """
    if not os.path.exists(checkpoint_dir):
        logging.error(f"Checkpoint directory {checkpoint_dir} does not exist!")
        return None
    checkpoint_files = [os.path.join(checkpoint_dir, f) 
                        for f in os.listdir(checkpoint_dir) 
                        if f.startswith('checkpoint_epoch_')]
    if not checkpoint_files:
        logging.error("No checkpoint files found!")
        return None
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    logging.info(f"Loading checkpoint from {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return latest_checkpoint

def generate_random_equation(max_digit=1, max_depth=2):
    """生成随机算式及其解题步骤"""
    ast = calc_ast.generate_random_ast(
        max_depth=random.randint(1, max_depth),
        max_digit=max_digit
    )
    question = calc_ast.ast_to_string(ast) + "="
    result, steps = calc_ast.calculate_steps(ast)
    full_equation = f"@{'|'.join(steps)}@={result}"
    return question, full_equation

def main():
    # 参数配置
    max_length = 256
    max_digit = 1      # 最大数字位数
    max_depth = 2       # 表达式最大深度
    num_samples = 10    # 生成样本数量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    end_token = "$"

    # 初始化模型和词汇表
    vocab = calculator_vocab.CalculatorVocab()
    end_token_idx = vocab.vocab_to_idx[end_token]

    # 创建模型实例
    model = calculator_model.TransformerDecoderModel(
        vocab=vocab,
        embed_size=128,
        num_heads=4,
        hidden_dim=2048,
        num_layers=8,
        max_length=max_length
    ).to(device)

    # 加载训练好的模型
    checkpoint_dir = "checkpoints"
    if not load_latest_checkpoint(checkpoint_dir, model, device):
        print("Failed to load model checkpoint!")
        return

    # 生成并推理随机算式
    print(f"Generating {num_samples} samples:")
    for _ in range(num_samples):
        # 生成随机算式
        question, ground_truth = generate_random_equation(max_digit, max_depth)
        print("\n" + "="*50)
        print(f"[Question] {question}")
        print(f"[Ground Truth] {ground_truth}")

        # 准备输入序列
        result = question
        input_str = question  # 初始输入为问题部分

        # 自回归生成过程
        print("[Model Generation]", end=" ")
        for _ in range(max_length - len(question)):
            # 编码当前序列
            input_ids = vocab.encode(
                input_str, 
                max_length=max_length,
                pad=True,
                add_end=False
            )
            input_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(device)

            # 模型推理
            with torch.no_grad():
                output = model(input_tensor)
            
            # 获取下一个token
            next_token_idx = output[0, len(input_str)-1].argmax().item()
            
            # 遇到结束标记则停止生成
            if next_token_idx == end_token_idx:
                break
            
            # 转换token并追加结果
            next_char = vocab.idx_to_vocab[next_token_idx]
            result += next_char
            input_str += next_char  # 更新输入序列
            print(next_char, end="", flush=True)
            time.sleep(0.05)
        print("\n[Model Result] " + result.split("=")[-1])

if __name__ == "__main__":
    main()