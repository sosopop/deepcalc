import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        """
        使用 register_buffer 注册位置编码，保证模型转移设备时自动跟随
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        参数:
            x: (batch_size, seq_len, d_model)
        返回:
            加上位置编码的 x
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab, embed_size, num_heads, hidden_dim, num_layers, max_length):
        """
        参数:
            vocab: 词汇表
            embed_size: 嵌入维度
            num_heads: 多头注意力头数
            hidden_dim: 前馈网络隐藏层维度
            num_layers: Transformer decoder 层数
            max_length: 序列最大长度（用于位置编码和预生成 mask）
            pad_idx: 填充符索引
        """
        super().__init__()
        self.embed_size = embed_size
        self.max_length = max_length
        self.num_heads = num_heads
        self.vocab = vocab
        self.pad_idx = vocab.vocab_to_idx[vocab.pad_token]

        # 嵌入层（设置 padding_idx 使得填充时自动忽略）
        self.embedding = nn.Embedding(self.vocab.vocab_size, embed_size, padding_idx=self.pad_idx)
        # 使用模块化的位置编码
        self.pos_encoder = PositionalEncoding(embed_size, max_len=max_length)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出映射层
        self.fc = nn.Linear(embed_size, self.vocab.vocab_size)

    def generate_causal_mask(self, tgt):
        batch_size, seq_len = tgt.size()
        device = tgt.device

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, dtype=torch.bool, device=device)
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size * self.num_heads, -1, -1)  # (batch_size * num_heads, seq_len, seq_len)
        
        return causal_mask

    def forward(self, tgt):
        """
        参数:
            tgt: (batch_size, seq_len) 目标序列的索引
        返回:
            (batch_size, seq_len, vocab_size) 模型输出 logits
        """
        
        causal_mask = self.generate_causal_mask(tgt)
        
        # 填充掩码：True 表示该位置为填充
        tgt_padding_mask = (tgt == self.pad_idx)

        # 嵌入并缩放（乘以 sqrt(embed_size)）
        tgt_emb = self.embedding(tgt) * math.sqrt(self.embed_size)
        tgt_emb = self.pos_encoder(tgt_emb)

        output = self.transformer_encoder(src=tgt_emb, mask=causal_mask, src_key_padding_mask=tgt_padding_mask)
        output = self.fc(output)
        return output


# 示例用法
if __name__ == '__main__':
    # 假设一些参数
    embed_size = 128
    num_heads = 8
    hidden_dim = 1024
    num_layers = 2
    max_length = 50
    pad_idx = 19          # 假设填充符索引为 19
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    import calculator_vocab
    import calculator_dataset_ast_reason as calculator_dataset
    vocab = calculator_vocab.CalculatorVocab()
    num_samples = 10
    max_digit = 2  # 测试时使用较小位数方便观察
    max_length = 256
    
    dataset = calculator_dataset.CalculatorDataset(num_samples, max_length, max_digit, max_depth=2, vocab=vocab)
    dataloader = calculator_dataset.DataLoader(dataset, batch_size=8, shuffle=False)
    # 创建模型
    model = TransformerDecoderModel(vocab, embed_size, num_heads, hidden_dim, num_layers, max_length)
    model.to(device)
    model.train()  # 训练模式下，tgt_mask 会生效
    
    for batch, batch_str in dataloader:
        batch = batch.to(device)
        for b, s in zip(batch, batch_str):
            print(s)
            print(vocab.decode(b.tolist(), remove_special=False))

        tgt = batch
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        

        output = model(tgt)
        print("训练模式下，模型输出形状：", output.shape)
