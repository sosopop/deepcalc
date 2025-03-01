import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Iterable, Optional, Tuple

    
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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # 线性变换矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        # 线性投影并分头
        Q = self.W_q(x).view(*x.shape[:2], self.num_heads, -1).permute(0, 2, 1, 3)
        K = self.W_k(x).view(*x.shape[:2], self.num_heads, -1).permute(0, 2, 1, 3)
        V = self.W_v(x).view(*x.shape[:2], self.num_heads, -1).permute(0, 2, 1, 3)
        a = F.scaled_dot_product_attention(Q, K, V, is_causal=mask is not None)
        o = a.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.W_o(o)

class SelfAttentionBlock(nn.Module):
    """单层自注意力 + 前馈网络"""
    def __init__(self, model_dim, num_heads, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(model_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, model_dim)
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 注意力子层
        attn_output = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # 前馈子层
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x

class SequentialProcessor(nn.Module):
    """堆叠多个处理层"""
    def __init__(self, num_blocks, model_dim, num_heads, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(model_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        
    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x

class CalculatorModel(nn.Module):
    """自回归序列生成模型"""
    def __init__(self, vocab, embed_dim, num_heads, ff_dim, num_blocks, max_seq_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.vocab = vocab
        self.pad_idx = vocab.vocab_to_idx[vocab.pad_token]

        # 输入表示层
        self.token_embed = nn.Embedding(vocab.vocab_size, embed_dim, padding_idx=self.pad_idx)
        self.position_enc = PositionalEncoding(embed_dim, max_len=max_seq_len)
        
        # 核心处理模块
        self.feature_processor = SequentialProcessor(
            num_blocks = num_blocks,
            model_dim = embed_dim,
            num_heads = num_heads,
            ff_dim = ff_dim
        )
        
        # 输出映射
        self.output_proj = nn.Linear(embed_dim, vocab.vocab_size)

    def _create_causal_mask(self, seq):
        _, seq_len = seq.size()
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=seq.device),
            diagonal=1
        )

    def forward(self, input_seq):
        # 生成掩码
        causal_mask = self._create_causal_mask(input_seq)
        
        # 构建输入表示
        embedded = self.token_embed(input_seq) * math.sqrt(self.embed_dim)
        position_aware = self.position_enc(embedded)
        
        # 特征处理
        processed = self.feature_processor(position_aware, causal_mask)
        
        # 生成输出
        return self.output_proj(processed)

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
    model = CalculatorModel(vocab, embed_size, num_heads, hidden_dim, num_layers, max_length)
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
