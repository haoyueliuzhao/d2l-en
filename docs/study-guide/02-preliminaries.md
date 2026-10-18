# 第 2 章：预备知识——把数学、张量与梯度连接起来

[返回导读目录](README.md) · [教材原章](../../chapter_preliminaries/index.md) · [重点术语表](../GLOSSARY.zh-CN.md)

本章建立后续所有模型的共同语言：数据怎样存储，运算如何对应数学，损失如何产生梯度，以及怎样描述不确定性。先修只需基本 Python；第 3 章训练循环会同时使用这里的形状、矩阵乘法、梯度与随机采样。阅读优先目标是能解释每个中间张量，而不是记住全部 API。

## 按教材目录逐节解析

### 2.1 数据操作（Data Manipulation）

[源文：ndarray.md](../../chapter_preliminaries/ndarray.md)。张量（tensor）既存储数值，也携带形状、类型和设备信息。重点比较 `reshape`、切片、拼接与逐元素操作：它们改变的对象不同。广播机制（broadcasting）从末尾维度对齐，每对维度必须相等或其中一个为 1；`(n, 1)` 与 `(n,)` 相减会得到 `(n, n)`，这一合法运算可能恰好是错误损失的来源。PyTorch 与 NumPy 某些转换共享内存，原位修改前要明确别名关系。

### 2.2 数据预处理（Data Preprocessing）

[源文：pandas.md](../../chapter_preliminaries/pandas.md)。用小型房屋 CSV 演示读取、分离输入与标签、处理缺失值、编码类别和转为张量。数值缺失填均值与类别缺失另设指示项表达不同假设；不要给无序类别随意编号后就当连续数值使用。用于评估的预处理统计应从训练部分学习，这一原则在后面房价交叉验证中尤其关键。

### 2.3 线性代数（Linear Algebra）

[源文：linear-algebra.md](../../chapter_preliminaries/linear-algebra.md)。标量、向量、矩阵和高阶张量构成形状语言。归约操作（reduction）沿指定轴汇总，`keepdim` 保留轴以方便广播；累积和不等于把整轴压缩为一个数。点积（dot product）、哈达玛积（Hadamard product）与矩阵乘法（matrix multiplication）必须分别理解。范数（norm）度量数值大小，后面会用于正则化与梯度检查。

### 2.4 微积分（Calculus）

[源文：calculus.md](../../chapter_preliminaries/calculus.md)。导数是局部变化率，偏导数（partial derivative）固定其他变量，梯度（gradient）把所有偏导数组成向量。小步负梯度方向常用于下降，但有限步长不保证损失降低。链式法则（chain rule）说明复合函数的依赖如何传递；绘图辅助函数只是呈现工具，不必先背下它们的内部实现。

### 2.5 自动微分（Automatic Differentiation）

[源文：autograd.md](../../chapter_preliminaries/autograd.md)。先跟踪 $y=2x^\top x$ 的梯度，再理解非标量输出、分离计算和 Python 控制流。自动微分（automatic differentiation）依据实际计算图（computational graph）应用链式法则；它既不是逐坐标扰动的数值差分，也不是先求全局符号表达式。PyTorch 的梯度默认累积，重复反向传播前应按目的清零；`detach` 截断一条依赖路径，但不让剩余路径全部失去梯度。

### 2.6 概率与统计（Probability and Statistics）

[源文：probability.md](../../chapter_preliminaries/probability.md)。抛硬币示例连接概率模型、抽样、统计估计与收敛。随后区分事件和随机变量（random variable），联合概率（joint probability）、条件概率（conditional probability）、独立与条件独立，再讨论期望（expectation）和不确定性。检测示例的重点是条件方向和基础发生率，不能把 $P(A\mid B)$ 当作 $P(B\mid A)$。更多数据可以减少参数估计的不确定性，但不必然消除结果自身的随机性。

### 2.7 查阅文档（Documentation）

[源文：lookup-api.md](../../chapter_preliminaries/lookup-api.md)。`dir` 用于发现成员，`help` 用于理解签名、参数和返回值。学习遇到形状或参数问题时，应定位具体运算的接口约定；不同框架的近似同名函数不保证轴语义、默认归约或原位行为完全一致。

## 三组需要能口算的关系

$$
A\in\mathbb R^{m\times d},\quad B\in\mathbb R^{d\times k}
\ \Longrightarrow\ AB\in\mathbb R^{m\times k},\qquad
(AB)_{ij}=\sum_{r=1}^{d}A_{ir}B_{rj}.
$$

内侧维度 $d$ 是求和轴；与同形矩阵逐元素相乘的含义不同。对批量数据，先决定哪一维是样本，再解释矩阵乘法。

$$
f(x)=2x^\top x\quad\Longrightarrow\quad\nabla_xf=4x,
\qquad \nabla_x\sum_j y_j(x)=\sum_j\nabla_xy_j(x).
$$

第二式解释了用全 1 上游梯度对向量输出反传的结果：得到各输出梯度之和，并不是完整雅可比矩阵（Jacobian）。

$$
P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)},\quad P(B)>0;
\qquad \mathbb E[X]=\sum_x xP(X=x).
$$

第一式是贝叶斯定理（Bayes' theorem）；第二式针对离散随机变量，连续情形使用积分。条件概率的分母是在已知条件下重新归一化，不可省略。

## 代码阅读路线与易错点

依次在 `ndarray.md` 检查 `(3,1)+(1,2)` 的广播，在 `linear-algebra.md` 比较 `sum`、`cumsum`、点积与矩阵乘法，然后完整读 `autograd.md` 的 `requires_grad_`、`backward`、`grad.zero_` 和 `detach` 示例。最后把 `pandas.md` 的每一列如何转成数值写出来，再转入训练模型。

`id` 变化只直接反映 Python 对象身份，不足以单独证明底层存储是否共享；判断内存还要结合运算语义。原位操作也不能一律视为优化，它可能覆盖反向传播所需的值。教材展示多框架分支；在现有 PyTorch 笔记本中沿同一分支阅读即可。

## 自测与回答线索

1. `(4,1)` 减 `(4,)` 为什么不是四个残差？线索：末尾维度广播后得到 `4×4`。
2. 为什么沿列归一化时常要保留归约轴？线索：检查分母形状能否按预期广播。
3. `u=x.detach()` 后计算 `u*x`，对 `x` 的导数是什么？线索：把 `u` 当常量，剩下乘法的另一条依赖。
4. 高准确度检测为什么不直接等于高阳性可信度？线索：条件概率方向不同，还需要基础发生率。

## 小实验建议（未执行）

- 用同一组数分别执行矩阵乘法、逐元素乘法和广播相减，先手写预期形状再核对；保存输入、输出和解释，不用训练网络。
- 对 $2x^\top x$ 比较手算、自动微分和有限差分。变化差分步长，记录误差；过大步长的近似误差与过小步长的舍入误差是待观察现象，不是本文的实测结论。
