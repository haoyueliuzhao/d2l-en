# 第 4 章：线性神经网络分类——从分数到概率，再到可靠评估

[返回导读目录](README.md) · [教材原章](../../chapter_linear-classification/index.md) · [重点术语表](../GLOSSARY.zh-CN.md)

本章把第 3 章的数值预测换成类别预测：数据加载和训练器基本复用，输出、损失和指标重新设计。先修是张量归约、概率与训练循环；后续多层感知机、卷积网络及许多语言模型都沿用这里的交叉熵。核心能力是解释概率如何产生、梯度为何合理，以及测试成绩在什么条件下可信。

## 按教材目录逐节解析

### 4.1 softmax 回归（Softmax Regression）

[源文：softmax-regression.md](../../chapter_linear-classification/softmax-regression.md)。多分类不能把无序类别编号当连续目标做普通回归。模型为每类产生一个未归一化分数（logit），softmax 将分数变成总和为 1 的概率；独热编码（one-hot encoding）用于数学表达，实际标签可以保留整数索引。交叉熵（cross-entropy）来自真实类别的负对数似然，信息论中的熵与“惊讶程度”提供另一种理解。

### 4.2 图像分类数据集（The Image Classification Dataset）

[源文：image-classification-dataset.md](../../chapter_linear-classification/image-classification-dataset.md)。Fashion-MNIST 有 10 类服装，原始图像为 `28×28` 灰度图；训练集 60,000 张，官方测试集 10,000 张。PyTorch 批量形状为 `(N,C,H,W)`，输入变换可以改变分辨率。先可视化标签和图像，再检查加载速度与打乱策略。源码把官方测试部分放入 `self.val` 供教学观察；正式调参应另从训练数据划出验证集，保留官方测试集作最终评估。

### 4.3 分类模型基类（The Base Classification Model）

[源文：classification.md](../../chapter_linear-classification/classification.md)。`Classifier` 在 `Module` 基础上增加准确率和验证报告。准确率（accuracy）比较最大分数或概率对应的类别与标签，不等于训练时优化的可微损失。原文指出，按批次等权平均会在最后一批大小不同时产生近似；精确数据集指标应按样本计数汇总。

### 4.4 从零实现（Softmax Regression Implementation from Scratch）

[源文：softmax-regression-scratch.md](../../chapter_linear-classification/softmax-regression-scratch.md)。先沿类别轴计算 softmax，再把 `28×28` 图像展平为 784 维向量。`cross_entropy` 用样本索引与标签索引取出真实类别概率，避免 Python 循环。观察错分图像比只看总准确率更容易发现相似类别和标签问题；展平同时放弃了空间邻接关系，后面 CNN 将改变这一点。

### 4.5 简洁实现（Concise Implementation of Softmax Regression）

[源文：softmax-regression-concise.md](../../chapter_linear-classification/softmax-regression-concise.md)。`Flatten → LazyLinear` 输出原始分数，`F.cross_entropy` 内部处理稳定的对数 softmax。不要在它之前再次 softmax：这样会把概率当分数再归一化，改变训练目标。理解指数溢出与对数下溢，才能知道简洁接口解决了什么问题。

### 4.6 分类中的泛化（Generalization in Classification）

[源文：generalization-classification.md](../../chapter_linear-classification/generalization-classification.md)。固定分类器的测试错误率是伯努利变量的样本平均，可以用统计界讨论估计误差。反复看同一测试集选择模型，会使最终模型依赖测试数据，从而破坏原先的独立性条件。统计学习理论讨论模型族与复杂度的控制，不能把某一次高分直接当总体性能的证明。

### 4.7 环境与分布偏移（Environment and Distribution Shift）

[源文：environment-and-distribution-shift.md](../../chapter_linear-classification/environment-and-distribution-shift.md)。协变量偏移（covariate shift）改变输入分布但保持给定输入的标签规律；标签偏移（label shift）改变类别比例但保持给定类别的输入分布；概念偏移（concept shift）改变预测关系。纠正方法依赖这些假设，不能对任意部署差异都保证有效。后半节把学习问题放回批量、在线、主动学习与环境交互的背景，并提醒决策影响不能由单个预测指标完全表达。

## 核心公式：概率、损失和稳定性

对 $q$ 类、一个样本的分数 $o\in\mathbb R^q$，真实类别为 $y$：

$$
p_j=\frac{e^{o_j}}{\sum_{k=1}^q e^{o_k}},\qquad
\ell(o,y)=-\log p_y,\qquad
\frac{\partial\ell}{\partial o_j}=p_j-\mathbf1\{j=y\}.
$$

梯度推动真实类概率上升，并压低其他类分数。所有分数同时加一个常数不会改变概率，因此可令 $m=\max_k o_k$，使用

$$
\ell(o,y)=\log\sum_k e^{o_k-m}+m-o_y.
$$

该形式避免先计算巨大指数。交叉熵还区分“勉强正确”和“高度自信正确”；准确率只看最终类别，两者不应混淆。

若分类器固定、测试样本独立同分布（independent and identically distributed），错误率 $\hat\epsilon$ 对总体错误率 $\epsilon$ 有两侧 Hoeffding 界：

$$
P(|\hat\epsilon-\epsilon|\ge t)\le 2e^{-2nt^2}.
$$

这里 $n$ 是测试样本数，$t>0$ 是允许误差。该界是本节单侧界的常见两侧整理，不描述发生分布偏移后的错误，也不能直接用于反复按测试集挑选的最终模型。

协变量偏移下的重要性权重为 $\beta(x)=p_T(x)/p_S(x)$。要求源分布在目标分布需要的区域具有覆盖，且条件标签规律保持不变；极端权重会增大估计方差。

## 代码路线与自测

依次读 `FashionMNIST.get_dataloader`、`Classifier.accuracy`、从零版 `softmax/cross_entropy` 和简洁版 `Classifier.loss`。在每一步标出 `(N,784) → (N,10) → 标量损失`，并区分输入、概率与整数标签。

1. softmax 为什么不改变分数的最大值位置？线索：指数单调、分母相同。
2. 对错误类别极度自信时，为什么交叉熵很大？线索：真实类别概率趋近 0，负对数增大。
3. 99% 准确率能否直接说明部署表现？线索：检查数据划分、类别比例、样本量和目标分布。
4. 标签比例变化一定是协变量偏移吗？线索：要分别检查保持不变的是哪个条件分布。

## 小实验建议（未执行）

- 构造分数 `[1000, 1001, 1002]`，比较直接指数实现和稳定对数 softmax 的有限值情况；同时确认两种数学表达在温和输入下相符。
- 固定训练集与验证集，统计线性模型的混淆矩阵（confusion matrix）和各类准确率，再检查主要错分图像。本文未运行训练，也未提供任何精度结论。
