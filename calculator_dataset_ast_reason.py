import torch
from torch.utils.data import Dataset, DataLoader
import calculator_vocab
import calculator_ast as calc_ast
import random

class CalculatorDataset(Dataset):
    def __init__(self, num_samples, max_length, max_digit, max_depth, vocab):
        self.num_samples = num_samples
        self.max_digit = max_digit
        self.vocab = vocab
        self.max_length = max_length
        self.max_depth = max_depth

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        exp_ast = calc_ast.generate_random_ast(max_depth=random.randint(1, self.max_depth), max_digit=self.max_digit)
        exp_str = calc_ast.ast_to_string(exp_ast)
        result, steps = calc_ast.calculate_steps(exp_ast)
        exp_str = f"{exp_str}=@{'|'.join(steps)}@={result}"
        encoded = self.vocab.encode(exp_str, max_length=self.max_length, pad=True)
        return torch.tensor(encoded, dtype=torch.long), exp_str

# 测试代码
if __name__ == '__main__':
    vocab = calculator_vocab.CalculatorVocab()
    num_samples = 10
    max_digit = 2  # 测试时使用较小位数方便观察
    max_length = 256
    
    dataset = CalculatorDataset(num_samples, max_length, max_digit, max_depth=3, vocab=vocab)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for batch, batch_str in dataloader:
        for b, s in zip(batch, batch_str):
            print(s)
            print(vocab.decode(b.tolist(), remove_special=False))
