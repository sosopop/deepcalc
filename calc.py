#!/usr/bin/env python
import os
import torch
import logging
import calculator_vocab
import calculator_model
import time

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

def main():
    # 参数设置（应与训练时保持一致）
    max_length = 256
    embed_size = 128
    num_heads = 4
    num_layers = 8
    hidden_dim = 2048

    # 设置运行设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载词汇表
    vocab = calculator_vocab.CalculatorVocab()

    # 实例化模型
    model = calculator_model.CalculatorModel(
        vocab,
        embed_size,
        num_heads,
        hidden_dim,
        num_layers,
        max_length
    )

    # 加载最近的 checkpoint
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        logging.error("Checkpoint directory does not exist!")
        return
    if load_latest_checkpoint(checkpoint_dir, model, device) is None:
        logging.error("Failed to load any checkpoint. Exiting.")
        return

    # 获取结束标记对应的索引
    end_token_idx = vocab.vocab_to_idx[vocab.end_token]

    # 提示用户输入算式
    print("请输入算式（例如 '1+1='），输入 'quit' 退出。")
    while True:
        user_input = input("算式: ").strip()
        if user_input.lower() == "quit":
            break
        if '=' not in user_input:
            print("错误：输入算式必须包含 '='")
            continue

        # 截取问号部分（包括 '='）
        # question = user_input[:user_input.index('=')+1]
        question = user_input
        result = question

        print("开始推理：")
        # 自回归生成过程：最多生成 (max_length - len(question)) 个字符
        for _ in range(max_length - len(question)):
            # 将当前结果编码为固定长度的 token 序列
            input_ids = vocab.encode(result, max_length=max_length, pad=True, add_end=False)
            input_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(device)  # (1, max_length)
            with torch.no_grad():
                output = model(input_tensor)  # 输出形状 (1, seq_len, vocab_size)
            # 取当前位置的 logits，生成下一个 token
            next_token = output[0, len(result)-1].argmax().item()
            if next_token == end_token_idx:
                break
            # 将生成的 token 转为字符，并追加到结果中
            c = vocab.idx_to_vocab[next_token]
            result += c
            print(c, end='', flush=True)
            time.sleep(0.2)

        print()
        result = result[:result.index('=')+1] + result[result.rindex('=')+1:]
        print("最终结果:", result)

if __name__ == "__main__":
    main()
