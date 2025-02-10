class CalculatorVocab:
    def __init__(self):
        """
        初始化计算器词汇表，其中:
          '$' 为结束符,
          '#' 为填充符.
        """
        self.end_token = '$'
        self.pad_token = '#'
        # 将结束符和填充符放到词汇表最后
        self.vocab = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '.', '+', '-', '*', '/', '(', ')', '=',
            self.end_token, self.pad_token
        ]
        self.vocab_to_idx = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.idx_to_vocab = {idx: ch for idx, ch in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def encode(self, text, max_length=None, pad=True, add_end=True):
        """
        将输入字符串编码成索引序列。

        参数:
            text (str): 待编码的字符串。
            max_length (int, optional): 序列的最大长度。如果提供该参数，当序列不足时（pad=True）会进行填充，
                                        或者超过时直接截断。
            pad (bool): 是否对不足 max_length 的序列进行填充，填充符为 '#'。

        返回:
            List[int]: 编码后的索引列表。

        说明:
            1. 编码时会自动在序列末尾添加结束符 '$'（如果输入中没有的话）。
            2. 如果文本中包含不在词汇表中的字符，则会抛出 ValueError 异常。
        """
        # 将字符串拆分为单个字符列表
        tokens = list(text)
        if add_end:
            # 如果末尾没有结束符，则自动添加
            if not tokens or tokens[-1] != self.end_token:
                tokens.append(self.end_token)

        # 将字符转换为对应的索引
        encoded = []
        for token in tokens:
            if token not in self.vocab_to_idx:
                raise ValueError(f"Token '{token}' 不在词汇表中！")
            encoded.append(self.vocab_to_idx[token])

        # 根据 max_length 参数进行截断或填充
        if max_length is not None:
            if len(encoded) < max_length and pad:
                pad_idx = self.vocab_to_idx[self.pad_token]
                encoded += [pad_idx] * (max_length - len(encoded))
            elif len(encoded) > max_length:
                encoded = encoded[:max_length]

        return encoded

    def decode(self, indices, remove_special=True):
        """
        将索引序列解码成字符串。

        参数:
            indices (List[int]): 待解码的索引列表。
            remove_special (bool): 是否自动去掉结束符 '$' 和填充符 '#'，默认为 True。

        返回:
            str: 解码后的字符串。
        """
        tokens = [self.idx_to_vocab[idx] for idx in indices]
        if remove_special:
            tokens = [token for token in tokens if token not in (self.end_token, self.pad_token)]
        return ''.join(tokens)


# 示例用法
if __name__ == '__main__':
    vocab_obj = CalculatorVocab()

    # 示例文本
    text = "3+5*2"
    print("原始文本:", text)

    # 编码：传入最大长度 10，且进行填充
    encoded = vocab_obj.encode(text, max_length=10, pad=True)
    print("编码后:", encoded)

    # 解码：自动去除结束符和填充符
    decoded = vocab_obj.decode(encoded, remove_special=True)
    print("解码后:", decoded)
