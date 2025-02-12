# calculator_dataset.py

import random
import torch
from torch.utils.data import Dataset, DataLoader
import calculator_vocab  # 请确保 calculator_vocab.py 与本文件在同一目录下

import random

def generate_random_number_str(max_digit, max_digit_ratio=0.7):
    """
    随机生成一个数字字符串，位数在 [1, max_digit] 之间。
    多位数的首位不为 '0'。
    
    参数:
        max_digit: 最大位数
        max_digit_ratio: 控制生成最大位数数字所占比例的参数。
                         当 max_digit_ratio > 0 时：
                           有 max_digit_ratio 的概率生成 max_digit 位数字，
                           其余 (1 - max_digit_ratio) 的概率在 1 到 max_digit-1 位之间均匀采样。
                         当 max_digit_ratio == 0 时：
                           在 1 到 max_digit 之间均匀采样（不使用比例控制策略）。
    """
    # 如果只有1位，则直接生成
    if max_digit == 1:
        return str(random.randint(0, 9))
    
    # 根据 max_digit_ratio 决定生成数字的位数
    if max_digit_ratio == 0:
        # 均匀采样 1 到 max_digit 之间的数字位数
        num_digits = random.randint(1, max_digit)
    else:
        # 按比例策略：以 max_digit_ratio 的概率生成 max_digit 位数字，
        # 否则在 1 到 max_digit-1 之间均匀采样
        if random.random() < max_digit_ratio:
            num_digits = max_digit
        else:
            num_digits = random.randint(1, max_digit - 1)
    
    # 根据位数生成随机数字：多位数的首位不能为 '0'
    if num_digits == 1:
        return str(random.randint(0, 9))
    else:
        first_digit = str(random.randint(1, 9))
        other_digits = ''.join(str(random.randint(0, 9)) for _ in range(num_digits - 1))
        return first_digit + other_digits

def add_with_steps(a_str, b_str):
    """
    简化后的中间步骤格式示例：
    "19+19=8130=38"
    符号说明：
    ' 表示进位，; 分隔步骤，= 开头表示最终结果
    """
    max_len = max(len(a_str), len(b_str))
    a_padded = a_str.zfill(max_len)
    b_padded = b_str.zfill(max_len)
    a_rev = a_padded[::-1]
    b_rev = b_padded[::-1]
    
    carry = 0
    steps = []
    current_digits = []
    
    for i in range(max_len):
        digit_a = int(a_rev[i])
        digit_b = int(b_rev[i])
        total = digit_a + digit_b + carry
        current_digit = total % 10
        new_carry = total // 10
        
        # 简化步骤：只显示当前位计算结果和进位
        steps.append(f"{current_digit}{new_carry}")
        current_digits.append(str(current_digit))
        carry = new_carry
    
    if carry > 0:
        current_digits.append(str(carry))
    
    current_digits.reverse()
    result_str = ''.join(current_digits)
    # 使用单竖线分隔步骤，最终结果前加等号
    return f"{a_str}+{b_str}={''.join(steps)}={result_str}"

class CalculatorDataset(Dataset):
    def __init__(self, num_samples, max_length, max_digit, vocab):
        self.num_samples = num_samples
        self.max_digit = max_digit
        self.vocab = vocab

        # 重新计算最小长度
        # 原式长度: len(a)+1+len(b) (如"19+19=")
        # 步骤部分: max_digit*2 (如"81") 
        # 结果部分: 1+1+max_digit+1 (如"=38$")
        min_length = (2*max_digit + 2) + 3*max_digit + (1 + max_digit + 1 + 1)
        
        if max_length < min_length:
            print(f"自动调整max_length为{min_length}")
            self.max_length = min_length
        else:
            self.max_length = max_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        a = generate_random_number_str(self.max_digit)
        b = generate_random_number_str(self.max_digit)
        eq_str = add_with_steps(a, b)
        
        encoded = self.vocab.encode(eq_str, max_length=self.max_length, pad=True)
        return torch.tensor(encoded, dtype=torch.long), eq_str

# 测试代码
if __name__ == '__main__':
    # 创建词汇表对象（定义在 calculator_vocab.py 中）
    vocab_obj = calculator_vocab.CalculatorVocab()

    # 定义数据集参数
    num_samples = 100       # 数据集中样本总数
    max_digit = 10          # 每个操作数最多 50 位，支持超长大数
    max_length = 128

    # 创建数据集
    dataset = CalculatorDataset(num_samples, max_length, max_digit, vocab_obj)
    # 使用 DataLoader 生成批量数据
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 遍历一个 batch，打印输出
    for batch, batch_str in dataloader:
        print("一个 batch 的数据（形状为 [batch_size, max_length]）：")
        print(batch)
        print("第一个样本对应的字符串：")
        print(batch_str)
        break
