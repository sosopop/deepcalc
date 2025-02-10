import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
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
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, max_length, pad_idx):
        """
        参数:
            vocab_size: 词汇表大小
            embed_size: 嵌入维度
            num_heads: 多头注意力头数
            hidden_dim: 前馈网络隐藏层维度
            num_layers: Transformer decoder 层数
            max_length: 序列最大长度（用于位置编码和预生成 mask）
            pad_idx: 填充符索引
        """
        super().__init__()
        self.embed_size = embed_size
        self.pad_idx = pad_idx
        self.max_length = max_length

        # 嵌入层（设置 padding_idx 使得填充时自动忽略）
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        # 使用模块化的位置编码
        self.pos_encoder = PositionalEncoding(embed_size, max_len=max_length)
        # 预生成后续 mask，shape: (max_length, max_length)
        self.register_buffer('tgt_mask_base', self.generate_square_subsequent_mask(max_length))

        # 构造 Transformer decoder 层（注意：使用 batch_first=True）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # 输出映射层
        self.fc = nn.Linear(embed_size, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        """
        生成后续 mask，禁止当前时刻“看到”未来信息
        返回 shape (sz, sz)，其中上三角部分（不含对角线）为 -inf，其他位置为 0
        """
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).bool()
        return mask

    def forward(self, tgt):
        """
        参数:
            tgt: (batch_size, seq_len) 目标序列的索引
        返回:
            (batch_size, seq_len, vocab_size) 模型输出 logits
        """
        batch_size, seq_len = tgt.size()
        # 填充掩码：True 表示该位置为填充
        tgt_padding_mask = (tgt == self.pad_idx)

        # 嵌入并缩放（乘以 sqrt(embed_size)）
        tgt_emb = self.embedding(tgt) * math.sqrt(self.embed_size)
        tgt_emb = self.pos_encoder(tgt_emb)

        # 只在训练时使用 tgt_mask，推理时允许模型自由生成输出
        # if self.training:
        tgt_mask = self.tgt_mask_base[:seq_len, :seq_len]
        # else:
        #     tgt_mask = None

        # 使用预生成的 memory 基础张量（全零），根据当前序列长度切片并扩展 batch_size
        memory = torch.zeros(batch_size, seq_len, self.embed_size, device=tgt.device)

        # Transformer decoder（注意：由于设置了 batch_first=True，无需转置）
        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        output = self.fc(output)
        return output


# 示例用法
if __name__ == '__main__':
    # 假设一些参数
    vocab_size = 20       # 示例词汇表大小
    embed_size = 128
    num_heads = 4
    hidden_dim = 256
    num_layers = 2
    max_length = 50
    pad_idx = 19          # 假设填充符索引为 19

    # 创建模型
    model = TransformerDecoderModel(vocab_size, embed_size, num_heads, hidden_dim, num_layers, max_length, pad_idx)
    model.train()  # 训练模式下，tgt_mask 会生效

    # 假设一个目标序列（batch_size=2, seq_len=10），其中部分位置可能为 pad_idx
    tgt = torch.randint(0, vocab_size - 2, (2, 10))
    # 模型输出 (2, 10, vocab_size)
    output = model(tgt)
    print("训练模式下，模型输出形状：", output.shape)

    # 切换到推理模式
    model.eval()
    output = model(tgt)
    print("推理模式下，模型输出形状：", output.shape)
