import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        scale = (self.d_model // self.num_heads) ** -0.5  # 显式计算缩放因子
        w = F.scaled_dot_product_attention(
            Q, K, V, 
            attn_mask=mask,
            is_causal=mask is not None, 
            scale=scale)
        
        # 具体实现
        # n_batch, n_ctx, n_state = x.shape
        # scale = 1 /  ((n_state // self.num_heads) ** 0.5)
        # qk = torch.matmul(Q, K.transpose(-2, -1)) * scale
        # if mask is not None:
        #     qk = qk + mask[:n_ctx, :n_ctx]
        # qk = qk.float()
        # w = F.softmax(qk, dim=-1).to(x.dtype)
        # w = w @ V
        
        o = w.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.W_o(o)

class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    
class MoE(nn.Module):
    """Mixture of Experts层"""
    def __init__(self, model_dim, num_experts, ff_dim, top_k=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, ff_dim),
                nn.GELU(),
                nn.Linear(ff_dim, model_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)])
        
        # 门控网络
        self.gate = nn.Linear(model_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)
        self.load_balance_loss = None  # 存储当前层的负载平衡损失

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.model_dim)  # (batch*seq, dim)
        
        # 计算门控权重
        gate_logits = self.gate(x_flat)  # (batch*seq, num_experts)
        gate_probs = self.softmax(gate_logits)
        
        # 选择top-k专家
        topk_weights, topk_indices = gate_probs.topk(self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # 初始化输出
        output = torch.zeros_like(x_flat)
        
        # 计算负载平衡损失
        importance = gate_probs.sum(0)  # (num_experts)
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        
        # 遍历所有专家计算输出
        for expert_idx in range(self.num_experts):
            # 找出当前专家被选中的位置（直接定位原始索引）
            mask = (topk_indices == expert_idx)
            selected_positions = mask.any(dim=-1)
            
            if not selected_positions.any():
                continue
                
            # 获取对应的行列索引
            rows, cols = mask.nonzero(as_tuple=True)
            
            # 直接使用原始索引获取权重和输入
            expert_input = x_flat[rows]
            expert_output = self.experts[expert_idx](expert_input)
            
            # 获取对应位置的权重
            weights = topk_weights[rows, cols].unsqueeze(-1)  # (selected, 1)
            
            # 累加输出
            output.index_add_(0, rows, weights * expert_output)
            
            # 统计专家使用情况
            expert_counts[expert_idx] = selected_positions.sum()
        
        # 计算负载平衡损失（基于Switch Transformer的实现）
        expert_fraction = expert_counts / (batch_size * seq_len * self.top_k)
        importance_fraction = importance / (batch_size * seq_len)
        self.load_balance_loss = self.num_experts * (expert_fraction * importance_fraction).sum()
        
        return output.view(batch_size, seq_len, -1)

class SelfAttentionBlock(nn.Module):
    """带有MoE的单层自注意力模块"""
    def __init__(self, model_dim, num_heads, num_experts=4, ff_dim=2048, top_k=2, dropout=0.05):
        super().__init__()
        self.self_attn = MultiHeadAttention(model_dim, num_heads)
        self.moe = MoE(model_dim, num_experts, ff_dim, top_k, dropout)
        self.norm1 = DeepseekV3RMSNorm(model_dim)
        self.norm2 = DeepseekV3RMSNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力子层
        attn_output = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # MoE前馈子层
        ff_output = self.moe(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x

class SequentialProcessor(nn.Module):
    """堆叠多个处理层（支持MoE配置）"""
    def __init__(self, num_blocks, model_dim, num_heads, num_experts=4, ff_dim=2048, top_k=2, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                num_experts=num_experts,
                ff_dim=ff_dim,
                top_k=top_k,
                dropout=dropout
            ) for _ in range(num_blocks)
        ])
        
    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x

class CalculatorModel(nn.Module):
    """带MoE架构的自回归模型"""
    def __init__(self, vocab, embed_dim, num_heads, ff_dim, num_blocks, max_seq_len, num_experts=4, top_k=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.vocab = vocab
        self.pad_idx = vocab.vocab_to_idx[vocab.pad_token]
        self.num_experts = num_experts
        self.top_k = top_k

        # 输入表示层
        self.token_embed = nn.Embedding(vocab.vocab_size, embed_dim, padding_idx=self.pad_idx)
        self.position_enc = PositionalEncoding(embed_dim, max_len=max_seq_len)
        
        # 核心处理模块（使用MoE）
        self.feature_processor = SequentialProcessor(
            num_blocks=num_blocks,
            model_dim=embed_dim,
            num_heads=num_heads,
            num_experts=num_experts,
            ff_dim=ff_dim,
            top_k=top_k
        )
        
        # 输出映射
        self.output_proj = nn.Linear(embed_dim, vocab.vocab_size)
        
        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """参数初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # 门控网络初始化
        for module in self.modules():
            if isinstance(module, MoE):
                nn.init.normal_(module.gate.weight, std=1e-3)

    def _create_causal_mask(self, seq):
        _, seq_len = seq.size()
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=seq.device),
            diagonal=1
        )
    
    def collect_moe_losses(self):
        """收集所有MoE层的负载平衡损失"""
        moe_losses = []
        for module in self.modules():
            if isinstance(module, MoE) and module.load_balance_loss is not None:
                moe_losses.append(module.load_balance_loss)
        return torch.stack(moe_losses).mean() if moe_losses else torch.tensor(0.0)
        
    def forward(self, input_seq):
        # 生成掩码
        causal_mask = self._create_causal_mask(input_seq).to(input_seq.device)
        
        # 构建输入表示
        embedded = self.token_embed(input_seq)
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
