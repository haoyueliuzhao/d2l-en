# 第 5 章：多层感知机——表达能力、梯度与泛化一起考虑

[返回导读目录](README.md) · [教材原章](../../chapter_multilayer-perceptrons/index.md) · [重点术语表](../GLOSSARY.zh-CN.md)

本章通过隐藏层和非线性，把线性分类器扩展成多层感知机（multilayer perceptron，MLP）。先修是第 2–4 章的链式法则、训练循环和交叉熵。与后续 CNN、循环网络和 Transformer 的联系在于：架构可以变化，但梯度传递、初始化、训练与验证的区分仍然存在。

## 按教材目录逐节解析

### 5.1 多层感知机（Multilayer Perceptrons）

[源文：mlp.md](../../chapter_multilayer-perceptrons/mlp.md)。隐藏层通过学习中间表示，使模型能够表达输入之间的非线性关系。多个仿射变换如果不插入非线性，仍能合并为一个仿射变换。比较修正线性单元（rectified linear unit，ReLU）、sigmoid 和 tanh 的函数值与导数；表达能力增强不等于任意训练算法都能找到好参数。

### 5.2 MLP 的实现（Implementation of Multilayer Perceptrons）

[源文：mlp-implementation.md](../../chapter_multilayer-perceptrons/mlp-implementation.md)。从零版为 Fashion-MNIST 建立 784 输入、256 隐藏单元、10 输出的模型；需要两组权重与偏置。简洁版用 `Sequential` 组合层，复用分类损失和训练器。关注隐藏维度是设计选择，样本数由批量轴表达，两者不能混为一谈。

### 5.3 前向传播、反向传播与计算图（Forward Propagation, Backward Propagation, and Computational Graphs）

[源文：backprop.md](../../chapter_multilayer-perceptrons/backprop.md)。前向传播（forward propagation）计算中间值与目标，反向传播（backpropagation）按依赖的逆序计算梯度。参数同时影响数据损失与正则项时，两条路径贡献要相加。训练需要保存中间激活，因此与仅预测相比，内存开销不只多出一份梯度。

### 5.4 数值稳定性与初始化（Numerical Stability and Initialization）

[源文：numerical-stability-and-init.md](../../chapter_multilayer-perceptrons/numerical-stability-and-init.md)。多层雅可比相乘可能导致梯度消失（vanishing gradients）或梯度爆炸（exploding gradients）。相同初始化还会使同层单元保持对称，难以学习不同特征。Xavier 初始化（Xavier initialization）在简化的独立、零均值假设下平衡前向与反向方差，是尺度启发式，不是任意深度和激活下的稳定性保证。

### 5.5 深度学习中的泛化（Generalization in Deep Learning）

[源文：generalization-deep.md](../../chapter_multilayer-perceptrons/generalization-deep.md)。过参数化网络可能拟合训练数据乃至随机标签，因此参数数量本身不足以解释泛化。归纳偏置（inductive bias）、优化过程、数据结构和正则化共同影响结果。早停（early stopping）利用训练阶段的差异选择模型，停止点应由验证数据决定；“训练误差为零”既不能独立证明模型好，也不能独立证明它一定泛化差。

### 5.6 暂退法（Dropout）

[源文：dropout.md](../../chapter_multilayer-perceptrons/dropout.md)。训练时随机置零部分隐藏激活，对保留项缩放以维持单个激活的条件期望。理解 `dropout_layer` 的边界情况与 `DropoutMLPScratch.forward` 的训练模式分支，再读框架 `Dropout`。常规推断关闭暂退法（dropout）；`eval()` 改变层行为，而禁用梯度是另一个控制开关。

### 5.7 Kaggle 房价预测（Predicting House Prices on Kaggle）

[源文：kaggle-house-price.md](../../chapter_multilayer-perceptrons/kaggle-house-price.md)。重点是现实表格数据流程：移除标识列、标准化数值列、处理缺失、独热编码、构造 $K$ 折与生成提交文件。当前源码直接把价格取对数作为训练标签，报告验证对数均方误差；不要误读成直接对原始价格训练。最终对各折模型预测先取指数再平均，平均价格与先平均对数再取指数通常不相同。

## 关键公式与直觉

对批量输入 $X\in\mathbb R^{n\times d}$，隐藏宽度 $h$，输出类别数 $q$：

$$
H=\phi(XW_1+b_1),\qquad O=HW_2+b_2,
\quad W_1\in\mathbb R^{d\times h},\ W_2\in\mathbb R^{h\times q}.
$$

$\phi$ 按元素作用；如果删掉它，两层可以折叠成 $X(W_1W_2)+(b_1W_2+b_2)$。非线性决定额外层能否超越线性模型族。

对单个样本，采用列向量约定 $z=W_1x$、$h=\phi(z)$、$o=W_2h$，若输出处上游梯度为 $g_o$，则

$$
\nabla_{W_2}J=g_oh^\top+\lambda W_2,\qquad
\nabla_zJ=(W_2^\top g_o)\odot\phi'(z).
$$

这里 $J$ 含平方权重惩罚，$\lambda$ 为惩罚系数；$\odot$ 表示逐元素乘法。矩阵方向随批量行向量、单样本列向量约定改变，应先核对形状再记转置。

暂退法令 $m\sim\operatorname{Bernoulli}(1-p)$，对 $0\le p<1$：

$$
\tilde h=\frac{mh}{1-p},\qquad \mathbb E[\tilde h\mid h]=h.
$$

$p$ 是丢弃概率，不是保留概率；$p=1$ 需单独返回零，不能代入分母。该等式只保证单个激活期望，经过后续非线性后，整个网络的输出期望不必与关闭 dropout 的输出严格相同。

## 代码路线与需要辨明的实现简化

从 `MLPScratch.forward` 的矩阵形状出发，到 `backprop.md` 手工追踪一个权重的梯度，再看 `dropout_layer` 与训练模式切换，最后完整追踪 `KaggleHouse.preprocess → get_dataloader → k_fold → submission.csv`。

房价示例在划折前合并训练与竞赛测试特征计算预处理。这是源码采用的教学流程；严格的独立验证应在每折训练部分拟合均值、方差和编码器，再应用于验证部分，避免验证信息影响预处理。这个提醒来自静态代码阅读，本文没有比较两种流程的实测成绩。源码中 `val` 也可能指无标签的竞赛测试特征，应按其内容理解，不能仅凭变量名认定它就是可计算验证损失的数据。

## 自测与回答线索

1. 为什么叠加两个线性层不一定增加非线性表达能力？线索：展开矩阵乘法与偏置即可合并。
2. 全零权重初始化与零偏置初始化有什么不同？线索：前者容易保持隐藏单元对称；偏置为零可与随机权重配合。
3. `eval()` 能否代替 `no_grad()`？线索：一个影响模式敏感层，一个影响梯度记录。
4. 对数误差更关注什么？线索：$\log\hat y-\log y=\log(\hat y/y)$，描述比例而非绝对价格差。

## 小实验建议（未执行）

- 固定数据、层宽与随机种子，比较有无 ReLU 的两层模型；同时记录训练与验证损失，避免只凭训练拟合评价增加深度的作用。
- 对同一输入分别在训练和评估模式重复前向传播，观察 dropout 输出变化；再检查批量平均激活。期望保持是理论性质，有限重复次数下不要求样本均值精确相等。
