from enum import Enum
import random

# ================================ 词法分析部分 ================================
class TokenType(Enum):
    """标记类型枚举，定义所有可能的符号类型"""
    NUMBER = 'NUMBER'  # 数字
    PLUS = '+'         # 加号
    MINUS = '-'        # 减号/负号
    MUL = '*'          # 乘号
    DIV = '/'          # 除号
    LPAREN = '('       # 左括号
    RPAREN = ')'       # 右括号
    EQ = '='           # 等号
    EOF = 'EOF'        # 输入结束标记

class Token:
    """标记类，存储标记类型和值（只有数字需要存储具体值）"""
    def __init__(self, type, value=None):
        self.type = type    # 标记类型（TokenType枚举成员）
        self.value = value   # 标记值（仅NUMBER类型需要）

    def __repr__(self):
        return f'Token({self.type}, {self.value})'

class Lexer:
    """词法分析器，将输入字符串转换为标记流"""
    def __init__(self, text):
        self.text = text            # 输入字符串
        self.pos = 0                # 当前位置
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None  # 当前字符

    def advance(self):
        """移动指针到下一个字符"""
        self.pos += 1
        if self.pos < len(self.text):
            self.current_char = self.text[self.pos]
        else:
            self.current_char = None

    def skip_whitespace(self):
        """跳过空白字符"""
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def number(self):
        """提取连续的数字字符，处理整数和浮点数"""
        result = ''
        # 循环收集数字和小数点
        while self.current_char is not None and (self.current_char.isdigit() or self.current_char == '.'):
            result += self.current_char
            self.advance()
        if '.' in result:  # 浮点数
            raise ValueError('浮点数暂不支持')
        # 根据是否包含小数点返回不同数值类型
        return float(result) if '.' in result else int(result)

    def get_next_token(self):
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char.isdigit() or self.current_char == '.':
                return Token(TokenType.NUMBER, self.number())

            if self.current_char == '+':
                self.advance()
                return Token(TokenType.PLUS)
            elif self.current_char == '-':
                self.advance()
                return Token(TokenType.MINUS)
            elif self.current_char == '*':
                self.advance()
                return Token(TokenType.MUL)
            elif self.current_char == '/':
                self.advance()
                return Token(TokenType.DIV)
            elif self.current_char == '(':
                self.advance()
                return Token(TokenType.LPAREN)
            elif self.current_char == ')':
                self.advance()
                return Token(TokenType.RPAREN)
            elif self.current_char == '=':
                self.advance()
                return Token(TokenType.EQ)
            else:
                raise Exception(f'Invalid character: {self.current_char}')

        return Token(TokenType.EOF)

# ================================ 语法分析部分 ================================
class ASTNode:
    """抽象语法树节点基类（空类，用于类型标记）"""
    pass

class NumberNode(ASTNode):
    """数字节点（叶子节点）"""
    def __init__(self, value):
        self.value = value  # 存储数值

    def __repr__(self):
        return f'Number({self.value})'

class BinaryOpNode(ASTNode):
    """二元运算节点"""
    def __init__(self, left, op, right):
        self.left = left   # 左子树
        self.op = op      # 操作符类型
        self.right = right # 右子树

    def __repr__(self):
        return f'BinaryOp({self.left}, {self.op}, {self.right})'

class UnaryOpNode(ASTNode):
    """一元运算节点（主要用于负数）"""
    def __init__(self, op, operand):
        self.op = op       # 操作符（只有负号）
        self.operand = operand  # 操作对象

    def __repr__(self):
        return f'UnaryOp({self.op}, {self.operand})'

class Parser:
    """语法分析器（递归下降解析器）"""
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()  # 当前处理的标记

    def eat(self, token_type):
        """验证当前标记类型并获取下一个标记"""
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            raise Exception(f'意外的标记: {self.current_token}, 期望: {token_type}')

    def parse(self):
        """解析入口点"""
        return self.parse_expression()

    # 解析优先级层次：
    # expression -> additive EQ additive
    # additive   -> multiplicative (PLUS/MINUS multiplicative)*
    # multiplicative -> unary (MUL/DIV unary)*
    # unary       -> (PLUS/MINUS) unary | primary
    # primary     -> NUMBER | LPAREN expression RPAREN

    def parse_expression(self):
        """处理最低优先级的等号运算（如果有）"""
        node = self.parse_additive()  # 先解析左边表达式
        
        # 支持连续的等号运算（虽然数学上不合理，但按语法允许）
        while self.current_token.type == TokenType.EQ:
            op = self.current_token.type
            self.eat(TokenType.EQ)
            right = self.parse_additive()
            node = BinaryOpNode(node, op, right)
            
        return node

    def parse_additive(self):
        """处理加减法（优先级高于等号）"""
        node = self.parse_multiplicative()  # 先解析乘除
        
        # 循环处理连续的加减法
        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            op = self.current_token.type
            self.eat(op)
            right = self.parse_multiplicative()
            node = BinaryOpNode(node, op, right)
            
        return node

    def parse_multiplicative(self):
        """处理乘除法（优先级高于加减法）"""
        node = self.parse_unary()  # 先解析一元运算
        
        # 循环处理连续的乘除法
        while self.current_token.type in (TokenType.MUL, TokenType.DIV):
            op = self.current_token.type
            self.eat(op)
            right = self.parse_unary()
            node = BinaryOpNode(node, op, right)
            
        return node

    def parse_unary(self):
        """处理一元运算符（正负号）"""
        if self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            op = self.current_token.type
            self.eat(op)
            # 递归处理可能的多重符号（如 --5）
            return UnaryOpNode(op, self.parse_unary())
        else:
            return self.parse_primary()

    def parse_primary(self):
        """处理基本元素：数字或括号表达式"""
        token = self.current_token
        if token.type == TokenType.NUMBER:
            self.eat(TokenType.NUMBER)
            return NumberNode(token.value)
        elif token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            node = self.parse_expression()  # 递归解析括号内的表达式
            self.eat(TokenType.RPAREN)
            return node
        else:
            raise Exception(f'意外的标记: {token}')

def generate_random_number_str(max_digit, max_digit_ratio=1.0):
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
        return random.randint(0, 9)
    
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
        return random.randint(0, 9)
    else:
        first_digit = str(random.randint(1, 9))
        other_digits = ''.join(str(random.randint(0, 9)) for _ in range(num_digits - 1))
        return int(first_digit + other_digits)
    
# ================================ 随机生成AST ================================
def generate_random_ast(max_depth=3, int_prob=1.0, max_digit=4, max_digit_ratio=1.0):
    """生成随机的抽象语法树"""
    if max_depth <= 0:
        # 生成数字节点（80%整数，20%浮点数）
        if random.random() < int_prob:
            return NumberNode(generate_random_number_str(max_digit, max_digit_ratio))
        else:
            # 暂不支持浮点数，抛出异常
            raise ValueError('暂不支持浮点数')
            # return NumberNode(round(random.uniform(0, 1000), 1))
    else:
        choice = random.choices(
            ['unary', 'binary'],
            weights=[0.1, 0.9],
            # weights=[0.0, 1.0],
            k=1
        )[0]
        if choice == 'unary':
            return UnaryOpNode(TokenType.MINUS, generate_random_ast(random.randint(0, max_depth-1), max_digit=max_digit, max_digit_ratio=max_digit_ratio))
        else:
            # 选择二元运算符（排除等号）
            op = random.choice([
                TokenType.PLUS,
                TokenType.MINUS,
                TokenType.MUL,
                TokenType.DIV
            ])
            
            if op == TokenType.PLUS:
                left = generate_random_ast(random.randint(0, max_depth-1), int_prob, max_digit, 0.5)
                right = generate_random_ast(random.randint(0, max_depth-1), int_prob, max_digit, 0.5)
            elif op == TokenType.MINUS:
                left = generate_random_ast(random.randint(0, max_depth-1), int_prob, max_digit, 0.5)
                right = generate_random_ast(random.randint(0, max_depth-1), int_prob, max_digit, 0.2)
            elif op == TokenType.MUL:
                left = generate_random_ast(random.randint(0, max_depth-1), int_prob, 2 if max_digit > 2 else max_digit, 0.5)
                right = generate_random_ast(random.randint(0, max_depth-1), int_prob, 2 if max_digit > 2 else max_digit, 0.5)
            elif op == TokenType.DIV:
                left = generate_random_ast(random.randint(0, max_depth-1), int_prob, max_digit, 0.5)
                right = generate_random_ast(random.randint(0, max_depth-1), int_prob, max_digit, 0.01)
            return BinaryOpNode(left, op, right)

# ================================ AST转字符串 ================================
def ast_to_string(node):
    """将AST转换为带括号的表达式字符串"""
    def get_priority(node):
        """获取节点优先级"""
        if isinstance(node, BinaryOpNode):
            if node.op in (TokenType.PLUS, TokenType.MINUS):
                return 1
            if node.op in (TokenType.MUL, TokenType.DIV):
                return 2
            return 0
        if isinstance(node, UnaryOpNode):
            return 3
        return 4  # 数字节点

    def helper(node, parent_priority):
        """递归辅助函数"""
        if isinstance(node, NumberNode):
            return str(node.value)
            
        if isinstance(node, UnaryOpNode):
            current_priority = 3
            operand_str = helper(node.operand, current_priority)
            operand_priority = get_priority(node.operand)
            
            # 处理操作数括号
            if operand_priority < current_priority:
                operand_str = f'({operand_str})'
                
            s = f'-{operand_str}'
            
            # 处理当前节点括号
            if current_priority < parent_priority:
                return f'({s})'
            return s
            
        if isinstance(node, BinaryOpNode):
            op_type = node.op
            current_priority = get_priority(node)
            op_symbol = op_type.value

            # 处理左子树
            left_str = helper(node.left, current_priority)
            left_priority = get_priority(node.left)
            
            # 左子树需要括号的情况：
            if left_priority < current_priority:
                left_str = f'({left_str})'

            # 处理右子树
            right_str = helper(node.right, current_priority)
            right_priority = get_priority(node.right)
            
            # 右子树需要括号的情况：
            # 1. 优先级小于当前
            # 2. 相同优先级且是减法或除法（右结合）
            if (right_priority < current_priority) or \
               (right_priority == current_priority and op_type in [TokenType.MINUS, TokenType.DIV]) or \
                (right_priority == current_priority and isinstance(node.right, BinaryOpNode) and node.right.op == TokenType.DIV):
                right_str = f'({right_str})'

            s = f'{left_str}{op_symbol}{right_str}'

            return s

        raise ValueError('未知节点类型')

    return helper(node, 0)  # 初始父级优先级为最低

def _add_positive(a_pos, b_pos):
    carry = 0
    steps = []
    temp_a, temp_b = a_pos, b_pos
    while temp_a > 0 or temp_b > 0 or carry > 0:
        digit_a = temp_a % 10
        digit_b = temp_b % 10
        total = digit_a + digit_b + carry
        current = total % 10
        new_carry = total // 10
        steps.append(f"{current}{new_carry}")
        carry = new_carry
        temp_a //= 10
        temp_b //= 10
    if not steps:
        steps.append("00")
    digits = [s[0] for s in steps]
    result = int(''.join(reversed(digits)).lstrip('0') or '0')
    return ';'.join(steps), result

def _sub_positive(a_pos, b_pos):
    steps = []
    result_digits = []
    borrow = 0
    temp_a, temp_b = a_pos, b_pos
    
    while temp_a > 0 or temp_b > 0:
        digit_a = temp_a % 10 - borrow
        borrow = 0
        digit_b = temp_b % 10
        
        if digit_a < digit_b:
            digit_a += 10
            borrow = 1
        current = digit_a - digit_b
        steps.append(f"{current}{borrow}")
        result_digits.append(str(current))
        temp_a //= 10
        temp_b //= 10
    
    result = int(''.join(reversed(result_digits)).lstrip('0') or '0')
    return ';'.join(reversed(steps)), int(result)

def add_with_steps(a, b):
    # 处理符号逻辑
    if (a < 0 and b < 0) or (a >= 0 and b >= 0):
        sign = -1 if a < 0 else 1
        steps_str, result_abs = _add_positive(abs(a), abs(b))
        result = sign * result_abs
    else:
        if abs(a) > abs(b):
            larger, smaller = abs(a), abs(b)
            sign = -1 if a < 0 else 1
        else:
            larger, smaller = abs(b), abs(a)
            sign = -1 if b < 0 else 1
        steps_str, result_abs = _sub_positive(larger, smaller)
        result = sign * result_abs
    
    return f"{a}+{b}={steps_str}={result}"

def sub_with_steps(a, b):

    # 处理符号逻辑
    if b < 0:
        return add_with_steps(a, -b)
    if a < 0:
        steps_str, res_abs = _add_positive(abs(a), b)
        return f"{a}-{b}={steps_str}={-res_abs}"
    
    if a >= b:
        steps_str, result = _sub_positive(a, b)
    else:
        steps_str, result_abs = _sub_positive(b, a)
        result = -result_abs
    
    return f"{a}-{b}={steps_str}={result}"
    
def mul_with_steps(a, b):
    def _mul_positive(a_pos, b_pos):
        partials = []
        temp_b = b_pos
        while temp_b > 0:
            b_digit = temp_b % 10
            temp_a = a_pos
            carry = 0
            while temp_a > 0 or carry > 0:
                a_digit = temp_a % 10
                product = a_digit * b_digit + carry
                partials.append(f"{product % 10}{product // 10}")
                carry = product // 10
                temp_a //= 10
            temp_b //= 10
        result = a_pos * b_pos
        return ';'.join(partials), result

    sign = -1 if (a < 0) ^ (b < 0) else 1
    steps_str, result_abs = _mul_positive(abs(a), abs(b))
    return f"{a}*{b}={steps_str}={sign * result_abs}"

def div_with_steps1(a, b):
    def _div_positive(a_pos, b_pos):
        steps = []
        current = 0
        quotient = []
        for digit in str(a_pos):
            current = current * 10 + int(digit)
            q = current // b_pos
            if q == 0:
                steps.append(f"0{current}")
            else:
                remainder = current % b_pos
                steps.append(f"{q}{remainder}")
                current = remainder
            quotient.append(str(q))
        quotient_str = ''.join(quotient).lstrip('0') or '0'
        return ';'.join(steps), int(quotient_str), current

    if b == 0:
        return f"{a}/0=E"

    sign = -1 if (a < 0) ^ (b < 0) else 1
    abs_a = abs(a)
    abs_b = abs(b)

    steps_str, quotient, remainder = _div_positive(abs_a, abs_b)

    # 调整商和余数以符合Python除法规则
    if remainder != 0 and sign == -1:
        quotient += 1
        adjustment_steps = []
        adjustment_steps.append(f"Adjust: {quotient-1}+1={quotient}")
        remainder = abs_b - remainder
        adjustment_steps.append(f"Remainder {abs_b - remainder}+{remainder}={abs_b}")
        steps_str += ";" + ";".join(adjustment_steps)

    final_quotient = sign * quotient
    return f"{a}/{b}={steps_str}={final_quotient}"

def div_with_steps(a, b):
    def _div_positive(a_pos, b_pos):
        steps = []
        current = 0
        quotient = []
        for digit in str(a_pos):
            current = current * 10 + int(digit)
            if current < b_pos:
                quotient.append('0')
                steps.append(f"0{current}")
            else:
                q = current // b_pos
                quotient.append(str(q))
                current = current % b_pos
                steps.append(f"{q}{current}")
        quotient_str = ''.join(quotient).lstrip('0') or '0'
        return ';'.join(steps), int(quotient_str), current

    if b == 0:
        return f"{a}/0=E"

    sign = -1 if (a < 0) ^ (b < 0) else 1
    abs_a, abs_b = abs(a), abs(b)
    steps_str, quotient_pos, remainder_pos = _div_positive(abs_a, abs_b)
    
    # 调整商以满足向下取整规则
    if remainder_pos != 0 and sign == -1:
        quotient_pos += 1  # 余数存在且结果为负时，商需+1
    
    final_quotient = sign * quotient_pos
    return f"{a}/{b}={steps_str}={final_quotient}"

def calculate_steps(node):
    """计算AST并记录计算步骤，忽略优先级括号，仅当一元操作数为负数时记录步骤"""
    steps = []

    def _evaluate(n):
        nonlocal steps
        if isinstance(n, NumberNode):
            return n.value, str(n.value)

        elif isinstance(n, UnaryOpNode):
            # 递归计算操作数的值和表达式
            operand_val, operand_expr = _evaluate(n.operand)
            current_val = -operand_val
            expr = f"-{operand_expr}"

            # 仅当操作数为负数时记录步骤
            if operand_val < 0:
                steps.append(f"{expr}={current_val}")
                return current_val, str(current_val)
            else:
                return current_val, expr

        elif isinstance(n, BinaryOpNode):
            # 递归计算左右子树
            left_val, left_expr = _evaluate(n.left)
            right_val, right_expr = _evaluate(n.right)
            op_symbol = n.op.value

            # 拼接表达式（不添加括号）
            expr = f"{left_expr}{op_symbol}{right_expr}"

            # 计算结果值
            if n.op == TokenType.PLUS:
                expr = add_with_steps(left_val, right_val)
                current_val = left_val + right_val
            elif n.op == TokenType.MINUS:
                expr = sub_with_steps(left_val, right_val)
                current_val = left_val - right_val
            elif n.op == TokenType.MUL:
                expr = mul_with_steps(left_val, right_val)
                current_val = left_val * right_val
            elif n.op == TokenType.DIV:
                # 检查除数是否为零
                if right_val == 0:
                    error_expr = f"{left_expr}/{right_expr}=E"
                    steps.append(error_expr)
                    raise ZeroDivisionError("Division by zero")
                expr = div_with_steps(left_val, right_val)
                current_val = left_val // right_val

            # 记录所有二元运算步骤
            # steps.append(f"{expr}={current_val}")
            steps.append(expr)
            return current_val, str(current_val)

        else:
            raise ValueError("未知节点类型")

    try:
        final_value, _ = _evaluate(node)
        return final_value, steps
    except ZeroDivisionError:
        # 捕获除零错误，返回'E'作为结果
        return 'E', steps

if __name__ == '__main__':
    # 测试用例
    test_cases = [
        '8/-9',
        '-9/5',
        '(8+4*5)/7',
        '(6933-7631)*(8595/5412)',
        "--(3+--5)*-2",
        "5*-3",
        "((2+3)*4)",
    ]

    for expr in test_cases:
        lexer = Lexer(expr)
        parser = Parser(lexer)
        ast = parser.parse()
        print(f"原式: {expr}")
        print(f"语法树重生成：{ast_to_string(ast)}")
        lexer = Lexer(ast_to_string(ast))
        parser = Parser(lexer)
        ast = parser.parse()
        result, steps = calculate_steps(ast)
        print(f"计算步骤: {steps}")
        print(f"最终结果: {result}\n")

    # 测试随机生成的表达式
    for _ in range(1000):
        ast = generate_random_ast(max_depth=2, max_digit=1)
        expr = ast_to_string(ast)
        print(f"随机生成的表达式: {expr}")
        lexer = Lexer(expr)
        parser = Parser(lexer)
        parsed_ast = parser.parse()
        result, steps = calculate_steps(ast)
        print(f"计算步骤: {steps}")
        print(f"结果: {result}\n")
        print(f"完整步骤: {expr}={'|'.join(steps)}\n")
        if result == 'E':
            try:
                assert result == eval(expr.replace('/','//'))
            except ZeroDivisionError:
                pass
        else:
            assert result == eval(expr.replace('/','//'))