import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import calculator_vocab
import calculator_model
import calculator_dataset_reason as calculator_dataset
# import calculator_dataset
import tqdm
import os
import logging

logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(model, vocab, device, train_loader, optimizer, criterion, epoch):
    model.train()
    epoch_loss = 0
    pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
    for batch_idx, (batch, _) in enumerate(pbar):
        tgt = batch
        tgt_input = tgt[:, :-1].to(device)
        tgt_output = tgt[:, 1:].to(device)

        optimizer.zero_grad()
        output = model(tgt_input)
        loss = criterion(output.view(-1, vocab.vocab_size), tgt_output.contiguous().view(-1))
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{epoch_loss / (pbar.n + 1):.5f}")
    
    return epoch_loss / len(train_loader)

def validate2(model, vocab, device, val_loader):
    model.eval()
    total = 0
    correct = 0
    end_token_idx = vocab.vocab_to_idx[vocab.end_token]
    
    with torch.no_grad():
        pbar = tqdm.tqdm(val_loader, desc='Validating', leave=False)
        for i, (batch, batch_str) in enumerate(pbar):
            for _, sample_str in zip(batch, batch_str):
                question = sample_str[0:sample_str.index('=')+1]
                result = question
                for _ in range(max_length - len(question)):
                    input = torch.LongTensor(vocab.encode(result, max_length=max_length, pad=True, add_end=False)).unsqueeze(0).to(device)
                    output = model(input)
                    next_token = output[0, len(result) - 1].argmax().item()
                    if next_token == end_token_idx:
                        break
                    result += vocab.idx_to_vocab[next_token]
                    
                if result == sample_str:
                    correct += 1
                total += 1
                
                pbar.set_postfix(accuracy=f"{correct/total:.5f}")
    
    accuracy = correct / total if total > 0 else 0
    return accuracy

def validate(model, vocab, device, val_loader):
    model.eval()
    total = 0
    correct = 0

    # 用于分别统计各运算符的正确数量和总样本数
    op_acc = {
        '+': {'correct': 0, 'total': 0},
        '-': {'correct': 0, 'total': 0},
        '*': {'correct': 0, 'total': 0},
        '/': {'correct': 0, 'total': 0},
    }

    end_token_idx = vocab.vocab_to_idx[vocab.end_token]
    equal_token_idx = vocab.vocab_to_idx['=']

    # 注意：val_loader 中每个 batch 返回的是 (tensor, [原始算式字符串列表])
    with torch.no_grad():
        pbar = tqdm.tqdm(val_loader, desc='Validating', leave=False)
        for batch, eq_str_list in pbar:
            batch = batch.to(device)
            tgt_input = batch[:, :-1]  # 模型输入（去掉最后一个 token）
            tgt_output = batch[:, 1:]  # 目标输出（去掉第一个 token）

            output = model(tgt_input)  # shape: (batch_size, seq_len-1, vocab_size)
            pred = output.argmax(dim=-1)  # 预测序列

            # 构造 mask：将 '=' 之前以及 end_token 之后的部分设为无效
            valid_mask = torch.ones_like(tgt_output, dtype=torch.bool)
            for i in range(batch.size(0)):
                eq_pos = (tgt_output[i] == equal_token_idx).nonzero(as_tuple=True)[0]
                end_pos = (tgt_output[i] == end_token_idx).nonzero(as_tuple=True)[0]
                if len(eq_pos) > 0:  # 如果存在 '='，则其之前的部分不计入预测
                    valid_mask[i, :eq_pos[0] + 1] = False
                if len(end_pos) > 0:  # 如果存在 end_token，则其之后的部分不计入预测
                    valid_mask[i, end_pos[0] + 1:] = False

            # 对于无效位置，不管预测是否正确，都视为正确
            matches = (pred == tgt_output) & valid_mask
            matches = matches == valid_mask  # 对于 mask=False 的位置，总是 True
            batch_correct = matches.all(dim=1)  # 每个样本是否所有有效位置均预测正确

            # 根据原始算式字符串判断运算符，统计各自的正确样本数量
            for i, eq in enumerate(eq_str_list):
                op_type = None
                # 此处简单地根据字符串中是否包含运算符判断类型
                for op in ['+', '-', '*', '/']:
                    if op in eq:
                        op_type = op
                        break
                if op_type is not None:
                    op_acc[op_type]['total'] += 1
                    if batch_correct[i].item():
                        op_acc[op_type]['correct'] += 1

            correct += batch_correct.sum().item()
            total += batch.size(0)
            pbar.set_postfix(accuracy=f"{correct/total:.4f}")

    overall_accuracy = correct / total if total > 0 else 0

    # 计算分别的准确率
    separate_accuracy = {}
    for op, counts in op_acc.items():
        if counts['total'] > 0:
            separate_accuracy[op] = counts['correct'] / counts['total']
        else:
            separate_accuracy[op] = None

    return overall_accuracy, separate_accuracy

def save_checkpoint(model, optimizer, epoch, loss, current_digits, accuracy, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_filename = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}_loss_{loss:.3f}_accuracy_{accuracy:.5f}_digits_{current_digits}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'current_digits': current_digits,
        'loss': loss,
        'accuracy': accuracy
    }, checkpoint_filename)
    return checkpoint_filename

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    current_digits = checkpoint['current_digits']
    accuracy = checkpoint['accuracy'] if 'accuracy' in checkpoint else 0.0
    loss = checkpoint['loss'] if 'loss' in checkpoint else 0.0
    return epoch, loss, current_digits, accuracy

if __name__ == '__main__':
    batch_size = 64
    max_length = 128
    num_samples = 100000
    max_digit = 20
    embed_size = 64
    num_heads = 8
    num_layers = 6
    hidden_dim = 2048
    learning_rate = 0.0001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab = calculator_vocab.CalculatorVocab()
    
    model = calculator_model.TransformerDecoderModel(vocab.vocab_size, 
                                                     embed_size=embed_size, 
                                                     num_heads=num_heads, 
                                                     hidden_dim=hidden_dim, 
                                                     num_layers=num_layers, 
                                                     max_length=max_length, 
                                                     pad_idx=vocab.vocab_to_idx[vocab.pad_token]).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.vocab_to_idx[vocab.pad_token])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    current_digits = 1
    start_epoch = 0
    loss = 0.0
    best_accuracy = 0.0
    checkpoint_dir = "checkpoints"
    
    if os.path.exists(checkpoint_dir):
        latest_checkpoint_path = max([os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')], key=os.path.getctime)
        if latest_checkpoint_path:
            start_epoch, loss, current_digits, best_accuracy = load_checkpoint(latest_checkpoint_path, model, optimizer)
            logging.info(f"Resuming from epoch {start_epoch}, loss {loss:.5f}, current_digits {current_digits}, best_accuracy {best_accuracy:.5f}")

    train_dataset = calculator_dataset.CalculatorDataset(num_samples, max_length, current_digits, vocab)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataset = calculator_dataset.CalculatorDataset(num_samples//10, max_length, current_digits, vocab)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logging.info("Example of training data:")
    for batch, batch_str in train_loader:
        for i in range(0, batch_size > 10 and 10 or batch_size):
            sample_str = vocab.decode(batch[i].tolist(), remove_special=False)
            logging.info(sample_str)
        break
    
    logging.info("Training...")
    for epoch in range(start_epoch, 1000):
        loss = train(model, vocab, device, train_loader, optimizer, criterion, epoch)
        overall_accuracy, separate_accuracy = validate(model, vocab, device, val_loader)
        logging.info(f"Epoch {epoch+1}: Loss={loss:.5f}, Overall Accuracy={overall_accuracy:.5f}, Current digits={current_digits}")
        for op, acc in separate_accuracy.items():
            if acc is not None:
                logging.info("Accuracy for {}: {:.4f}".format(op, acc))
            else:
                logging.info("Accuracy for {}: N/A (no samples)".format(op))

        if overall_accuracy > best_accuracy:
            if overall_accuracy == 1.0:
                if current_digits < max_digit:
                    current_digits += 1
                    logging.info(f"Increased digits to {current_digits}")
                    train_dataset = calculator_dataset.CalculatorDataset(num_samples, max_length, current_digits, vocab)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
                    val_dataset = calculator_dataset.CalculatorDataset(num_samples//10, max_length, current_digits, vocab)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                    best_accuracy = 0.0
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                    
                    logging.info("Example of training data:")
                    for batch, batch_str in train_loader:
                        for i in range(0, batch_size > 10 and 10 or batch_size):
                            sample_str = vocab.decode(batch[i].tolist(), remove_special=False)
                            logging.info(sample_str)
                        break
            else:
                best_accuracy = overall_accuracy
                
            checkpoint_filepath = save_checkpoint(model, optimizer, epoch, loss, current_digits, best_accuracy, checkpoint_dir)
            logging.info(f"Checkpoint saved: {checkpoint_filepath}")
        