import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import calculator_vocab
import calculator_model
import calculator_dataset_ast_reason as calculator_dataset
import tqdm
import os
import logging
from torch.amp import GradScaler

# 设置日志输出级别
logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(model, vocab, device, train_loader, optimizer, criterion, epoch):
    model.train()
    epoch_loss = 0
    eq_idx = vocab.vocab_to_idx['=']
    
    pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
    for batch_idx, (batch, _) in enumerate(pbar):
        tgt = batch
        tgt_input = tgt[:, :-1].to(device)
        tgt_output = tgt[:, 1:].to(device)

        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            output = model(tgt_input)
            # 找出每个样本中第一个等号的位置
            eq_positions = torch.argmax((tgt_output == eq_idx).int(), dim=1)  # (batch_size,)
            cols = torch.arange(tgt_output.size(1)).view(1, -1)  # (1, seq_len)
            cols = cols.expand(tgt_output.size(0), -1)  # (batch_size, seq_len)
            cols = cols.to(device)
            
            # 生成eq_mask矩阵：当列位置 > 等号位置时为True（需要屏蔽）
            eq_mask = cols > eq_positions.view(-1, 1) 
            
            # 计算每个位置的损失
            output_flat = output.view(-1, output.size(-1))  # (batch*seq, vocab)
            tgt_flat = tgt_output.contiguous().view(-1)      # (batch*seq)
            loss_per_token = criterion(output_flat, tgt_flat)
            
            # 应用eq_mask，仅保留需要计算损失的token
            mask_flat = eq_mask.view(-1)
            selected_loss = loss_per_token[mask_flat]
            
            # 计算平均损失，处理无有效损失的情况
            if selected_loss.numel() > 0:
                loss = selected_loss.mean()
            else:
                loss = torch.tensor(0.0, device=device)
        
        # loss.backward()
        # optimizer.step()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{epoch_loss / (pbar.n + 1):.5f}")
    
    return epoch_loss / len(train_loader)

def validate(model, vocab, device, val_loader):
    model.eval()
    total = 0
    correct = 0

    end_token_idx = vocab.vocab_to_idx[vocab.end_token]
    equal_token_idx = vocab.vocab_to_idx['=']

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

            correct += batch_correct.sum().item()
            total += batch.size(0)
            pbar.set_postfix(accuracy=f"{correct/total:.4f}")

    overall_accuracy = correct / total if total > 0 else 0
    return overall_accuracy

def save_checkpoint(model, optimizer, epoch, loss, current_digits, current_depth, accuracy, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_filename = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}_loss_{loss:.5f}_accuracy_{accuracy:.5f}_digits_{current_digits}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'current_digits': current_digits,
        'current_depth': current_depth,
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
    current_depth = checkpoint['current_depth']
    accuracy = checkpoint['accuracy'] if 'accuracy' in checkpoint else 0.0
    loss = checkpoint['loss'] if 'loss' in checkpoint else 0.0
    return epoch, loss, current_digits, current_depth, accuracy

if __name__ == '__main__':
    batch_size = 128
    max_length = 256
    num_samples = 100000
    max_digit = 20
    embed_size = 128
    num_heads = 4
    num_layers = 8
    hidden_dim = 2048
    learning_rate = 0.0001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = GradScaler(enabled=True)
    vocab = calculator_vocab.CalculatorVocab()
    model = calculator_model.CalculatorModel(vocab,
                                            embed_size,
                                            num_heads,
                                            hidden_dim,
                                            num_layers,
                                            max_length).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.vocab_to_idx[vocab.pad_token], reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    current_digits = 1
    current_depth = 1
    start_epoch = 0
    loss = 0.0
    best_accuracy = 0.0
    checkpoint_dir = "checkpoints"
    
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
        if len(checkpoint_files) > 0:
            latest_checkpoint_path = max(checkpoint_files, key=os.path.getctime)
            if latest_checkpoint_path:
                start_epoch, loss, current_digits, current_depth, best_accuracy = load_checkpoint(latest_checkpoint_path, model, optimizer)
                logging.info(f"Resuming from epoch {start_epoch}, loss {loss:.5f}, current_digits {current_digits}, current_depth {current_depth}, best_accuracy {best_accuracy:.5f}")

    train_dataset = calculator_dataset.CalculatorDataset(num_samples, max_length, current_digits, current_depth, vocab)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataset = calculator_dataset.CalculatorDataset(num_samples//10, max_length, current_digits, current_depth, vocab)
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
        accuracy = validate(model, vocab, device, val_loader)
        logging.info(f"Epoch {epoch+1}: Loss={loss:.5f}, Accuracy={accuracy:.5f}, Current digits={current_digits}")

        if accuracy > best_accuracy:
            if accuracy > 0.99:
                if current_digits < max_digit:
                    if current_depth <= current_digits:
                        current_depth += 1
                        logging.info(f"Increased depth to {current_depth}")
                    else:
                        current_digits += 1
                        logging.info(f"Increased digits to {current_digits}")
                    train_dataset = calculator_dataset.CalculatorDataset(num_samples, max_length, current_digits, current_depth, vocab)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
                    val_dataset = calculator_dataset.CalculatorDataset(num_samples//10, max_length, current_digits, current_depth, vocab)
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
                best_accuracy = accuracy
                
            checkpoint_filepath = save_checkpoint(model, optimizer, epoch, loss, current_digits, current_depth, best_accuracy, checkpoint_dir)
            logging.info(f"Checkpoint saved: {checkpoint_filepath}")
