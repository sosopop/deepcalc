# export.py
import torch
import torch.jit as jit
import os
import logging
import calculator_vocab
import calculator_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def export_model(checkpoint_dir, export_path, max_length=256):
    """
    导出优化后的TorchScript模型
    
    参数:
        checkpoint_dir: 检查点目录路径
        export_path: 导出模型路径（应以.pt结尾）
        max_length: 模型支持的最大序列长度
    """
    # 初始化词汇表和空模型
    vocab = calculator_vocab.CalculatorVocab()
    device = torch.device('cpu')
    
    # 查找最新检查点
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    latest_checkpoint = max(
        [os.path.join(checkpoint_dir, f) for f in checkpoints],
        key=os.path.getctime
    )
    logging.info(f"Loading checkpoint: {latest_checkpoint}")

    # 创建模型实例（参数需与训练时一致）
    model = calculator_model.TransformerDecoderModel(
        vocab=vocab,
        embed_size=128,
        num_heads=4,
        hidden_dim=2048,
        num_layers=8,
        max_length=max_length
    ).to(device)
    
    # 加载检查点
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 准备示例输入（符合模型预期的格式）
    example_input = torch.full((1, max_length-1), 
                              vocab.vocab_to_idx[vocab.pad_token], 
                              dtype=torch.long, 
                              device=device)
    
    # 使用JIT跟踪导出优化模型
    with torch.no_grad():
        traced_model = jit.trace(model, example_input, check_trace=False)
    
    # 保存模型和词汇表
    export_data = {
        'model': traced_model,
        'vocab': vocab,
        'config': {
            'max_length': max_length,
            'pad_idx': vocab.vocab_to_idx[vocab.pad_token],
            'end_idx': vocab.vocab_to_idx[vocab.end_token]
        }
    }
    # jit.save(export_data, export_path)
    
    jit.save(traced_model, export_path)
    logging.info(f"Model exported to {export_path}")

def test_exported_model(model_path):
    """
    测试导出的模型
    """
    # 加载导出的模型
    exported = jit.load(model_path)
    model = exported['model']
    vocab = exported['vocab']
    
    # 示例输入
    test_expression = "(12+3.5)*2="
    encoded = vocab.encode(test_expression, max_length=256, add_end=False)
    input_tensor = torch.tensor([encoded], dtype=torch.long)
    
    # 运行推理
    with torch.no_grad():
        output = model(input_tensor[:, :-1])
        predictions = output.argmax(dim=-1)
    
    # 解码输出
    decoded = vocab.decode(predictions[0].tolist())
    logging.info(f"Input: {test_expression}")
    logging.info(f"Output: {decoded}")

if __name__ == "__main__":
    # 配置参数
    CHECKPOINT_DIR = "checkpoints"
    EXPORT_PATH = "calculator_model.pt"
    
    # 执行导出
    export_model(CHECKPOINT_DIR, EXPORT_PATH)
    
    # 测试导出的模型
    logging.info("\nTesting exported model:")
    test_exported_model(EXPORT_PATH)