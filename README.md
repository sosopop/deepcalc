# 不精确计算器（Approximate Calculator）

基于Transformer解码器实现的数学表达式计算器，通过生成中间推理步骤提升计算准确率。本项目可作为学习Transformer架构和推理策略的实践案例。

![transformer](https://raw.githubusercontent.com/sosopop/deepcalc/main/asserts/logo.png) 

## 技术原理

### Transformer解码器架构
- 采用纯解码器结构
- 位置编码（Positional Encoding）捕获序列顺序
- 自注意力机制学习字符间长距离依赖
- 屏蔽注意力（Masked Attention）防止信息泄露

### 推理增强策略
模型通过生成中间计算步骤（思维链）来提升准确率：
```plaintext
输入： 6*5/(1-9)=@6*5=03;30=30|1-9=80=-8|30/-8=03;36=-3@=-4$
```
1. **分步计算**：将复杂运算分解为多个简单步骤
2. **中间验证**：每个步骤结果作为后续计算的上下文
3. **错误修正**：通过概率采样选择最优计算路径

## 项目结构
```
calculator-project/
├── calculator_ast.py         # 语法树生成模块
├── calculator_vocab.py       # 词汇表与编解码
├── calculator_model.py       # Transformer模型定义
├── calculator_dataset_ast_reason.py # 数据集生成
├── train.py                  # 训练主程序
├── calc.py                   # 计算器交互程序
└── test.py                   # 批量测试脚本
```

## 安装与使用

### 环境要求
```bash
pip install -r requirements.txt
```

### 训练模型
```bash
python train.py
```

### 交互式计算
```bash
python calc.py
请输入算式（例如 '1+1='），输入 'quit' 退出。
算式: 3*4+2=
开始推理：
@3*4=21;10=12|12+2=40;10=14@=14
最终结果: 3*4+2=14
```

### 批量测试
```bash
python test.py
[Question] (8-3)*(1+2)=
[Ground Truth] @8-3=50=5|1+2=30=3|5*3=51;10=15@=15
[Model Generation] @8-3=50=5|1+2=30=3|5*3=51;10=15@=15
[Model Result] 15
```

## 训练配置

### 渐进式训练策略

训练策略采用逐步训练，逐步增加模型复杂度，稳步学习复杂的多项式计算任务。
```python
# 在train.py中动态调整难度
if accuracy > 0.99:
    current_digits += 1  # 逐步增加数字位数
    current_depth += 1   # 逐步加深语法树层级
```

## 贡献指南

欢迎通过以下方式参与贡献：
1. 实现更多运算符（如^幂运算）
2. 优化语法树生成逻辑
3. 添加可视化中间步骤功能
4. 改进训练策略

请先fork项目并提交Pull Request

## 学习资源

通过本项目可以学习：
1. Transformer解码器的实现原理
2. 动态屏蔽注意力机制
3. 渐进式训练策略
4. 推理增强的实现方法
5. 语法树到序列的转换

⭐ **如果本项目对您有帮助，欢迎点击Star支持！** ⭐

*持续更新中，建议Watch关注项目进展*

