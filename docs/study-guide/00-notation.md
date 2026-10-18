# 符号篇解析：用形状和类型读懂公式

[返回解析目录](README.md) · [教材原文](../../chapter_notation/index.md) · [术语表](../GLOSSARY.zh-CN.md)

## 阅读原则

符号篇提供的是统一语言。读公式时先辨认对象的类型、形状和取值范围，再处理计算。小写粗体通常表示向量（vector），大写粗体通常表示矩阵（matrix）；但这些是排版约定，具体维度仍由上下文定义。$\mathbb R$ 则明确表示实数集合，不能当作任意变量名。

## 按原文顺序建立索引

| 部分 | 需要掌握的符号 | 学习重点 |
| --- | --- | --- |
| Numerical Objects | $x,\mathbf{x},\mathbf{X},\mathsf X,\mathbf I,x_i,x_{ij}$ | 区分标量、向量、矩阵和一般张量；下标说明取哪个元素，不自动说明该轴的业务含义。 |
| Set Theory | $\mathcal X,\mathbb Z,\mathbb R^n,\mathbb R^{a\times b},\lvert\mathcal X\rvert,\cup,\cap,\setminus$ | 先明确变量属于哪个集合。集合基数不同于向量范数，集合差也不同于数值减法。 |
| Functions and Operators | $f,\log,\exp,\mathbf 1,{}^\top,{}^{-1},\odot,\lVert\cdot\rVert_p,\langle\cdot,\cdot\rangle,\sum,\prod$ | 对每个运算核对输入、输出及成立条件。转置、逆矩阵、逐元素乘法和矩阵乘法不是可互换的操作。 |
| Calculus | $dy/dx,\partial y/\partial x,\nabla_{\mathbf x}y,\int$ | 普通导数、偏导数和梯度（gradient）回答不同维度上的变化问题。 |
| Probability and Information Theory | $X\sim P,P(X=x),P(X\mid Y),p,E,\operatorname{Var},\operatorname{Cov},H,D_{\mathrm{KL}}$ | 区分随机变量、实现值、分布、密度和统计量；不要把概率密度当成单点概率。 |

## 一个贯穿后续章节的形状例子

为避免行向量与列向量混淆，下面显式采用列向量约定：

$$
\mathbf X\in\mathbb R^{n\times d},\qquad
\mathbf w\in\mathbb R^{d\times1},\qquad
\mathbf y\in\mathbb R^{n\times1}.
$$

其中 $n$ 是样本数，$d$ 是特征数。线性预测为

$$
\hat{\mathbf y}=\mathbf X\mathbf w+b\mathbf 1_n,
\qquad
L=\frac{1}{2n}\lVert\hat{\mathbf y}-\mathbf y\rVert_2^2.
$$

标量偏置 $b$ 在实现中通常通过广播（broadcasting）加到每个样本。梯度为

$$
\nabla_{\mathbf w}L=\frac1n\mathbf X^{\top}(\hat{\mathbf y}-\mathbf y),\qquad
\frac{\partial L}{\partial b}=\frac1n\sum_{i=1}^{n}(\hat y_i-y_i).
$$

第一式输出必须是 $d\times1$，与被更新参数同形状；第二式必须是标量。先做形状检查，往往就能发现转置或归约轴错误。这里的 $1/2$ 是为抵消平方求导中的系数 2；若损失改为不带 $1/2$ 的均方误差，梯度系数也会改变。

一个常见代码陷阱是预测形状为 `(n, 1)`、标签形状为 `(n,)`。两者相减可能广播成 `(n, n)`，程序仍能执行，却不再是逐样本残差。数学符号里两者看起来都像“一个向量”，代码中必须显式对齐。

## 范数、概率与信息量的易错点

对 $p\geq1$，常用向量 $p$ 范数为

$$
\lVert\mathbf x\rVert_p=\left(\sum_i|x_i|^p\right)^{1/p}.
$$

它描述向量的大小，而维度描述坐标数量。矩阵元素逐项平方求和再开方是弗罗贝尼乌斯范数（Frobenius norm），不应自动等同于所有场合的矩阵谱范数。

连续随机变量在区间上的概率需要积分：$P(a\leq X\leq b)=\int_a^b p(x)\,dx$。密度 $p(x)$ 可以超过 1，只要相应积分满足概率规则。

离散分布的 KL 散度（KL divergence）应按公式辨认方向：

$$
D_{\mathrm{KL}}(P\Vert Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}.
$$

这里用 $P$ 加权，分子是 $P$，分母是 $Q$，通常不等于反向的 $D_{\mathrm{KL}}(Q\Vert P)$。若 $P(x)>0$ 而 $Q(x)=0$，相应散度可以为无穷大。避免只记“从哪个分布到哪个分布”的口头表述而丢失公式含义。默认自然对数给出以 nat 为单位的信息量；使用 $\log_2$ 时单位为 bit。

## 与代码的对应路线

先读 [线性代数](../../chapter_preliminaries/linear-algebra.md)，确认乘法与归约轴；再读 [微积分](../../chapter_preliminaries/calculus.md) 和 [自动微分](../../chapter_preliminaries/autograd.md)，把手算梯度对应到计算图；需要更系统的数学解释时回到 [数学附录](../../chapter_appendix-mathematics-for-deep-learning/index.md)。

## 自测与回答线索

1. **$\mathbf X\mathbf w$ 和 $\mathbf X\odot\mathbf w$ 是同一个计算吗？** 线索：前者沿共享维度求和，后者按元素相乘并受广播规则约束。
2. **梯度为什么需要与参数形状对应？** 线索：每个参数分量都需要一个局部变化率，才能逐分量更新。
3. **密度大于 1 为什么不必然违规？** 线索：连续概率由区间积分得到，不由一个点的密度值直接得到。
4. **为什么 KL 散度不能当作通常的对称距离？** 线索：交换分布会改变权重和对数比值。

## 建议练习（未执行）

用 $n=3,d=2$ 的小矩阵手算一次预测和梯度，随后用自动微分核对。再故意把标签从 `(3,1)` 改成 `(3,)`，只观察残差形状，解释为什么这种错误可能不触发异常。该练习的目的在于理解形状，而非测量模型性能。
