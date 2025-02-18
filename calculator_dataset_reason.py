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
    
    if len(a_str) == 1 and len(b_str) == 1:
        return f"{a_str}+{b_str}={result_str}"
    return f"{a_str}+{b_str}={';'.join(steps)}={result_str}"

def sub_with_steps(a_str, b_str):
    """
    生成减法的中间步骤字符串，格式示例："218-387=916010=-169"
    步骤说明：
    - 当被减数小于减数时，交换两者并添加负号
    - 步骤部分每两位表示当前位计算结果和借位
    """
    # 比较数值大小决定是否交换
    sign = 1
    if int(a_str) < int(b_str):
        a_str, b_str = b_str, a_str
        sign = -1

    max_len = max(len(a_str), len(b_str))
    a_padded = a_str.zfill(max_len)
    b_padded = b_str.zfill(max_len)
    a_rev = a_padded[::-1]
    b_rev = b_padded[::-1]
    
    borrow = 0
    steps = []
    current_digits = []
    
    for i in range(max_len):
        digit_a = int(a_rev[i])
        digit_b = int(b_rev[i])
        
        # 减去之前的借位
        digit_a -= borrow
        borrow = 0  # 重置借位
        
        if digit_a < digit_b:
            # 需要借位
            digit_a += 10
            borrow = 1
        
        current_digit = digit_a - digit_b
        steps.append(f"{current_digit}{borrow}")
        current_digits.append(str(current_digit))
    
    # 处理结果符号和数值
    current_digits.reverse()
    result_num = int(''.join(current_digits).lstrip('0') or '0')
    result_str = f"{result_num * sign}" if sign == -1 else str(result_num)
    
    # 拼接步骤字符串
    original_equation = f"{a_str}-{b_str}" if sign == 1 else f"{b_str}-{a_str}"
    
    if len(a_str) == 1 and len(b_str) == 1:
        return f"{original_equation}={result_str}"
    return f"{original_equation}={';'.join(steps)}={result_str}"

def mul_with_steps(a_str, b_str):
    """
    生成乘法的中间步骤字符串，格式示例：
    723*984=120828241656271863=252
    """
    reversed_a = a_str[::-1]  # 被乘数反转（从个位开始）
    reversed_b = b_str[::-1]  # 乘数反转（从个位开始）
    
    partials = []
    # 遍历乘数的每一位（从个位到最高位）
    for i in range(len(reversed_b)):
        b_digit = reversed_b[i]
        # 遍历被乘数的每一位（从个位到最高位）
        for j in range(len(reversed_a)):
            a_digit = reversed_a[j]
            # 计算当前位的乘积
            product = int(b_digit) * int(a_digit)
            # 计算需要补的零数（等于位权之和）
            # zeros = i + j
            # 生成中间结果字符串（保留所有数字和补零）
            # partial = f"{product}{'0' * zeros}"
            partial = f"{product:02d}"
            partials.append(partial)
    
    # 计算最终结果（所有中间结果求和）
    final_result = sum(int(p) for p in partials)
    
    # 拼接完整步骤字符串
    steps_str = ';'.join(partials)
    
    if len(a_str) == 1 and len(b_str) == 1:
        return f"{a_str}*{b_str}={final_result}"
    return f"{a_str}*{b_str}={steps_str}={final_result}"

def div_with_steps(a_str, b_str):
    """
    生成除法的中间步骤字符串，格式示例：
      "123/4=013003=30_3"
    算法说明（模拟除法）：
      - 从被除数的最高位开始依次取数，与除数比较；
      - 如果当前数字小于除数，则当前商位为 0，并记录 (0,当前余数)；
      - 否则，计算当前商位 q = 当前数字 // 除数，更新余数为 当前数字 - q * 除数，并记录 (q,余数)。
    最终结果为：若余数为 0，则为商；否则以 "商_余数" 的形式表示。
    """
    divisor = int(b_str)
    current = 0
    steps = []
    quotient_digits = []
    
    for digit in a_str:
        current = current * 10 + int(digit)
        if current < divisor:
            # 若已开始产生商位，则补0；否则直接记录当前余数
            if quotient_digits:
                quotient_digits.append('0')
            steps.append(f"0{current}")
        else:
            q = current // divisor
            quotient_digits.append(str(q))
            current = current - q * divisor
            steps.append(f"{q}{current}")
    
    # 处理被除数小于除数的情况
    if not quotient_digits:
        quotient_str = "0"
    else:
        quotient_str = ''.join(quotient_digits)
        # 去除可能的前导零
        quotient_str = str(int(quotient_str))
    
    final_result = quotient_str if current == 0 else quotient_str + '_' + str(current)
    steps_str = ';'.join(steps)
    
    if len(a_str) == 1 and len(b_str) == 1:
        return f"{a_str}/{b_str}={final_result}"
    return f"{a_str}/{b_str}={steps_str}={final_result}"

class CalculatorDataset(Dataset):
    def __init__(self, num_samples, max_length, max_digit, vocab):
        self.num_samples = num_samples
        self.max_digit = max_digit
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 随机选择运算符（扩展为四种：加、减、乘、除）
        op = random.choice(['+', '-', '*', '/'])
        
        
        if op == '-':
            a = generate_random_number_str(self.max_digit, max_digit_ratio=0.7)
            b = generate_random_number_str(self.max_digit, max_digit_ratio=0.7)
            eq_str = sub_with_steps(a, b)
        elif op == '*':
            a = generate_random_number_str(self.max_digit, max_digit_ratio=0.5)
            b = generate_random_number_str(self.max_digit, max_digit_ratio=0.5)
            eq_str = mul_with_steps(a, b)
        elif op == '/':
            a = generate_random_number_str(self.max_digit, max_digit_ratio=0.7)
            b = generate_random_number_str(self.max_digit, max_digit_ratio=0.1)
            # 确保除数 b 不为 0
            while b == "0":
                b = generate_random_number_str(self.max_digit, max_digit_ratio=0.1)
            eq_str = div_with_steps(a, b)
        else:  # op == '+'
            a = generate_random_number_str(self.max_digit, max_digit_ratio=0.7)
            b = generate_random_number_str(self.max_digit, max_digit_ratio=0.7)
            eq_str = add_with_steps(a, b)
        
        encoded = self.vocab.encode(eq_str, max_length=self.max_length, pad=True)
        return torch.tensor(encoded, dtype=torch.long), eq_str

# 测试代码
if __name__ == '__main__':
    vocab_obj = calculator_vocab.CalculatorVocab()
    num_samples = 100
    max_digit = 3  # 测试时使用较小位数方便观察
    max_length = 128
    
    dataset = CalculatorDataset(num_samples, max_length, max_digit, vocab_obj)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for batch, batch_str in dataloader:
        for s in batch_str:
            print(s)
