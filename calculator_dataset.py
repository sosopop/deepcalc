# calculator_dataset.py

import random
import torch
from torch.utils.data import Dataset, DataLoader
import calculator_vocab  # 请确保 calculator_vocab.py 与本文件在同一目录下
import gmpy2  # 第三方大数运算库，需提前安装： pip install gmpy2

def generate_random_number_str(max_digit):
    """
    随机生成一个数字字符串，位数在 [1, max_digit] 之间。
    多位数的首位不为 '0'。
    """
    num_digits = random.randint(1, max_digit)
    if num_digits == 1:
        return str(random.randint(0, 9))
    else:
        first_digit = str(random.randint(1, 9))
        other_digits = ''.join(str(random.randint(0, 9)) for _ in range(num_digits - 1))
        return first_digit + other_digits

def big_add(num1_str, num2_str):
    """
    使用 gmpy2 实现大数加法，不受 Python 内置整型长度限制。
    
    参数:
        num1_str, num2_str (str): 表示非负整数的字符串
    返回:
        str: 两数之和的字符串表示
    """
    a = gmpy2.mpz(num1_str)
    b = gmpy2.mpz(num2_str)
    result = a + b
    return str(result)

class CalculatorDataset(Dataset):
    def __init__(self, num_samples, max_length, max_digit, vocab):
        """
        参数:
            num_samples (int): 数据集样本数量
            max_length (int): 每条数据固定的最大长度（包括结束符和填充符）
            max_digit (int): 随机生成的整数的最大位数（每个操作数的最大位数）
            vocab (CalculatorVocab): 词汇表对象，用于文本编码

        注意: 当生成最大位数的操作数时，加法表达式的最小长度为 3*max_digit + 4，
              如果传入的 max_length 太小，则自动调整为该值。
        """
        self.num_samples = num_samples
        self.max_digit = max_digit
        self.vocab = vocab

        # 表达式格式为: a + b = result, 加上结束符。最小长度为:
        # len(a) + 1 ('+') + len(b) + 1 ('=') + len(result) + 1 (结束符)
        # 其中 a、b 最大为 max_digit 位，result 可能有 max_digit+1 位
        min_length = 3 * max_digit + 4  
        if max_length < min_length:
            print(f"Warning: provided max_length ({max_length}) is too short for max_digit={max_digit}. "
                  f"Automatically set max_length to {min_length}.")
            self.max_length = min_length
        else:
            self.max_length = max_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 使用大数生成函数生成两个随机大数（字符串形式）
        a = generate_random_number_str(self.max_digit)
        b = generate_random_number_str(self.max_digit)
        result = big_add(a, b)

        # 构造加法算式字符串，不包含结束符，后续 encode() 方法会自动添加结束符
        eq_str = f"{a}+{b}={result}"
        # 使用词汇表编码，自动在末尾添加结束符并填充到 self.max_length
        encoded = self.vocab.encode(eq_str, max_length=self.max_length, pad=True)
        # 返回 tensor 类型的索引序列
        return torch.tensor(encoded, dtype=torch.long), eq_str


# 测试代码
if __name__ == '__main__':
    # 创建词汇表对象（定义在 calculator_vocab.py 中）
    vocab_obj = calculator_vocab.CalculatorVocab()

    # 定义数据集参数
    num_samples = 100       # 数据集中样本总数
    max_digit = 50          # 每个操作数最多 50 位，支持超长大数
    # 根据最大位数计算表达式所需最小长度：3*max_digit + 4
    min_required_length = 3 * max_digit + 4
    max_length = min_required_length  # 也可设置为更大的固定长度

    # 创建数据集
    dataset = CalculatorDataset(num_samples, max_length, max_digit, vocab_obj)
    # 使用 DataLoader 生成批量数据
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 遍历一个 batch，打印输出
    for batch in dataloader:
        print("一个 batch 的数据（形状为 [batch_size, max_length]）：")
        print(batch)
        # 查看第一个样本对应的字符串（包含结束符和填充符）
        sample_str = vocab_obj.decode(batch[0].tolist(), remove_special=False)
        print("第一个样本对应的字符串：")
        print(sample_str)
        break
