import torch
from torch.utils.data import Dataset, DataLoader
import calculator_vocab
import calculator_ast as calc_ast

class CalculatorDataset(Dataset):
    def __init__(self, num_samples, max_length, max_digit, vocab):
        self.num_samples = num_samples
        self.max_digit = max_digit
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        exp_ast = calc_ast.generate_random_ast(max_depth=2)
        exp_str = calc_ast.ast_to_string(exp_ast)
        _, steps = calc_ast.calculate_steps(exp_ast)
        exp_str = f"{exp_str}={'|'.join(steps)}"
        encoded = self.vocab.encode(exp_str, max_length=self.max_length, pad=True)
        return torch.tensor(encoded, dtype=torch.long), exp_str

# 测试代码
if __name__ == '__main__':
    vocab_obj = calculator_vocab.CalculatorVocab()
    num_samples = 10
    max_digit = 3  # 测试时使用较小位数方便观察
    max_length = 128
    
    dataset = CalculatorDataset(num_samples, max_length, max_digit, vocab_obj)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for batch, batch_str in dataloader:
        for s in batch_str:
            print(s)
