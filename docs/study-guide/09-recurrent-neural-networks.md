# 第 9 章：循环神经网络重点解析

[学习导览](README.md) · [本章原文](../../chapter_recurrent-neural-networks/index.md) · [术语对照表](../GLOSSARY.zh-CN.md)

本章将独立样本建模推进到序列建模（sequence modeling）。先修是条件概率、矩阵乘法、交叉熵和计算图。重点是理解历史信息如何进入预测、同一组参数如何跨时间复用，以及为什么一步预测准确仍可能多步预测失败。

## 核心关系与公式

序列的概率链式分解为

$$p(x_1,\ldots,x_T)=\prod_{t=1}^{T}p(x_t\mid x_{<t}).$$

这是概率恒等式；只依赖最近 $\tau$ 个观测的马尔可夫假设（Markov assumption）则是一种建模近似。自回归（autoregressive）描述条件依赖方式，不局限于 RNN。

基本循环神经网络（recurrent neural network，RNN）的状态更新为

$$H_t=\tanh(X_tW_{xh}+H_{t-1}W_{hh}+b_h),\qquad O_t=H_tW_{hq}+b_q.$$

设批量为 $B$、输入维度为 $d$、隐藏维度为 $h$、词表大小为 $V$，则 $X_t\in\mathbb R^{B\times d}$，$H_t\in\mathbb R^{B\times h}$，$O_t\in\mathbb R^{B\times V}$。所有时间步共享权重，但各步状态不同。$O_t$ 是未归一化分数，不是概率。

以自然对数计算有效 token 的平均负对数似然 $L$，困惑度（perplexity）为

$$\operatorname{PPL}=\exp(L)=\exp\!\left(-\frac1N\sum_{t=1}^{N}\log p(x_t\mid x_{<t})\right).$$

$N$ 为参与评价的 token 数；比较困惑度要使用相同语料切分、分词方式和评估规则。字符级与词级困惑度不能直接排名。

## 按教材顺序解析

### 9.1 使用序列

[原文与代码](../../chapter_recurrent-neural-networks/sequence.md)。教材以带噪正弦序列构造固定窗口特征，把“前 $\tau$ 个值预测下一个值”转为监督学习。一步预测使用真实历史，多步滚动预测则把此前预测反馈为输入，误差和输入分布变化可能累积。按时间划分训练与后续评估，才能体现预测未来的任务。

### 9.2 将原始文本转换为序列数据

[原文与代码](../../chapter_recurrent-neural-networks/text-sequence.md)。流程是读取与清洗、分词（tokenization）、构建词表（vocabulary）、映射索引。教材用《时间机器》构造字符序列，并用词频统计展示自然语言的长尾现象。`<unk>` 表示未知 token；索引只是离散编码，不意味着相邻索引具有相近语义。清洗规则会丢弃原始信息，须视为任务定义的一部分。

### 9.3 语言模型

[原文与代码](../../chapter_recurrent-neural-networks/language-model.md)。语言模型（language model）估计下一 token 的条件分布，$n$ 元语法（n-gram）展示了频率估计、稀疏性与平滑的关系。困惑度衡量模型对真实序列的概率分配，不直接衡量事实正确性或文本实用性。构造样本时，标签是输入整体右移一位；注意原文的子序列划分说明与 `TimeMachine` 数据实现的具体切片，避免自行假定跨批状态连续。

### 9.4 循环神经网络

[原文与代码](../../chapter_recurrent-neural-networks/rnn.md)。隐藏状态（hidden state）使模型能够在固定维度的向量中总结过去。把时间轴展开后，RNN 是共享参数的前馈计算图；“存在循环连接”不意味着一次前向计算无法确定顺序。字符级模型将字符独热编码（one-hot encoding）输入 RNN，再将隐藏状态映射到词表分数。

### 9.5 从零实现 RNN

[原文与代码](../../chapter_recurrent-neural-networks/rnn-scratch.md)。先读 `RNNScratch` 如何逐步更新状态，再看 `RNNLMScratch` 如何组织输入与输出形状。代码内部常以 `(时间, 批量, 特征)` 处理序列，而数据加载器常给出 `(批量, 时间)` 的索引，转置不能省略。梯度裁剪（gradient clipping）对整组梯度的范数限幅，能缓解梯度爆炸，但不能恢复已经消失的梯度。生成时前缀用于预热状态，之后才循环生成新字符。

### 9.6 RNN 的简洁实现

[原文与代码](../../chapter_recurrent-neural-networks/rnn-concise.md)。框架层封装相同递推思想，并可采用优化内核。应逐一核对序列输出与最终状态的维度，尤其不要把所有时间步输出当成最后状态。对比从零实现时还要统一初始化、激活、参数布局和损失处理；仅同名“RNN”不足以保证数值完全相同。

### 9.7 通过时间反向传播

[原文](../../chapter_recurrent-neural-networks/bptt.md)。通过时间反向传播（backpropagation through time，BPTT）把共享参数在各时间位置的梯度贡献相加。状态雅可比（Jacobian）的长链乘积可能放大或压缩梯度；不是所有长序列都必然爆炸，也不能只看某一个权重元素就判断稳定性。截断（truncation）缩短梯度传播路径以限制计算和存储，但会损失远距离学习信号；保留状态的数值与保留完整计算图是两件事。

## 代码阅读路线

从 [序列窗口的 `Data`](../../chapter_recurrent-neural-networks/sequence.md) 出发，手工写出长度为 6、窗口为 3 时的输入与标签；随后读 [文本的 `TimeMachine` 与 `Vocab`](../../chapter_recurrent-neural-networks/text-sequence.md) 及 [语言模型中的数据批次构造](../../chapter_recurrent-neural-networks/language-model.md)。最后沿 [从零实现的 `one_hot` → `RNNScratch.forward` → `output_layer` → `clip_gradients` → `predict`](../../chapter_recurrent-neural-networks/rnn-scratch.md) 追踪一次训练和一次生成。

## 易错点与自测

- **窗口长度和隐藏维度分别控制什么？** 前者限制一次样本显式提供的序列长度，后者决定状态容量；二者不是同一个超参数。
- **为何训练标签必须右移？** 答案线索：若输入 token 与同位置标签相同，模型可能学习复制而非预测下一项。
- **裁剪后的梯度方向变了吗？** 对整个梯度向量做统一正比例缩放时方向不变；逐元素裁剪则不具备这一性质。
- **随机打乱的两个批次能直接承接状态吗？** 一般不能，样本的时间连续性已被破坏。若主动设计连续批次，还需决定何处切断计算图。
- **PPL 为 1 意味着什么？** 对所有评估 token 赋予概率 1 的理想极限；不是一般模型训练应达到的必然目标。

## 小实验建议（本次未执行）

1. 固定序列与训练划分，比较一步真实历史预测和 1、4、16 步滚动预测的误差。保存每个预测距离的均方误差，不将可视化中某一段较好拟合当作整体结论。
2. 固定模型与批量，比较两种裁剪阈值及不裁剪设置，记录裁剪前梯度范数、非有限值出现次数与验证困惑度。不同设置最好重复多个种子；这是对当前配置稳定性的观察，不是 RNN 全部性质的证明。
