import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import calculator_vocab
import calculator_model
import calculator_dataset
import tqdm

def train(model, vocab, device, train_loader, optimizer, criterion, epoch):
    model.train()
    epoch_loss = 0
    pbar = tqdm.tqdm(train_loader)
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
        pbar.update(1)

def validate(model, vocab, device, val_loader):
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
                
                pbar.set_postfix(accuracy=f"{correct/total:.4f}")
                pbar.update(1)
    
    accuracy = correct / total if total > 0 else 0
    print(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == '__main__':
    batch_size = 64
    max_length = 128
    num_samples = 100000
    max_digit = 7
    embed_size = 64
    num_heads = 8
    num_layers = 6
    hidden_dim = 2048
    learning_rate = 0.0002
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab = calculator_vocab.CalculatorVocab()
    train_dataset = calculator_dataset.CalculatorDataset(num_samples, max_length, max_digit, vocab)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataset = calculator_dataset.CalculatorDataset(1000, max_length, max_digit, vocab)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    print("Example of training data:")
    for batch, batch_str in train_loader:
        for i in range(0, batch_size > 10 and 10 or batch_size):
            sample_str = vocab.decode(batch[i].tolist(), remove_special=False)
            print(sample_str)
        break

    model = calculator_model.TransformerDecoderModel(vocab.vocab_size, 
                                                     embed_size=embed_size, 
                                                     num_heads=num_heads, 
                                                     hidden_dim=hidden_dim, 
                                                     num_layers=num_layers, 
                                                     max_length=max_length, 
                                                     pad_idx=vocab.vocab_to_idx[vocab.pad_token]).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.vocab_to_idx[vocab.pad_token])
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Training...")
    for epoch in range(10):
        train(model, vocab, device, train_loader, optimizer, criterion, epoch)
        validate(model, vocab, device, val_loader)