import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)
    
def rotate_half(x):
    """将输入张量的后半部分旋转"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, sin, cos):
    """应用旋转位置编码到query和key"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # 注册频率基（关键修改：确保维度正确性）
        self.register_buffer(
            "inv_freq",
            1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        )
        
        # 线性变换矩阵保持不变
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _get_rotary_matrix(self, seq_len, device):
        """生成旋转矩阵的sin/cos分量（修正维度问题）"""
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        
        # 关键修改：扩展频率到完整维度
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        freqs = freqs.repeat_interleave(2, dim=-1)  # 扩展维度到head_dim
        
        return torch.sin(freqs), torch.cos(freqs)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 线性投影并分头（保持原逻辑）
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 生成旋转矩阵（修正后的维度）
        sin, cos = self._get_rotary_matrix(seq_len, x.device)
        
        # 调整维度对齐（关键修改：正确广播维度）
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        
        # 应用旋转位置编码
        Q, K = apply_rotary_pos_emb(Q, K, sin, cos)
        
        # 缩放点积注意力（保持原逻辑）
        scale = self.head_dim ** -0.5
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            is_causal=mask is None,
            scale=scale
        )
        
        # 合并多头输出（保持原逻辑）
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        return self.W_o(attn_output)

class SelfAttentionBlock(nn.Module):
    """单层自注意力 + 前馈网络"""
    def __init__(self, model_dim, num_heads, ff_dim=2048, dropout=0.05):
        super().__init__()
        self.self_attn = MultiHeadAttention(model_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.SiLU(),
            nn.Linear(ff_dim, model_dim)
        )
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
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
    """自回归序列生成模型（使用RoPE）"""
    def __init__(self, vocab, embed_dim, num_heads, ff_dim, num_blocks, max_seq_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.vocab = vocab
        self.pad_idx = vocab.vocab_to_idx[vocab.pad_token]

        # 输入表示层（移除了位置编码）
        self.token_embed = nn.Embedding(vocab.vocab_size, embed_dim, padding_idx=self.pad_idx)
        
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
        embedded = self.token_embed(input_seq)
        
        # 特征处理（RoPE在注意力内部实现）
        processed = self.feature_processor(embedded, causal_mask)
        
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
