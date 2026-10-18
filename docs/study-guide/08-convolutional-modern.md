# 第 8 章：现代卷积神经网络重点解析

[学习导览](README.md) · [本章原文](../../chapter_convolutional-modern/index.md) · [术语对照表](../GLOSSARY.zh-CN.md)

本章的学习目标是解释“为什么这样连接层”，而非背诵各网络的层数。先修知识是卷积（convolution）、池化（pooling）、反向传播（backpropagation）与张量形状。读完应能比较不同模块如何控制参数量、计算量、特征复用和训练难度。本章沿用教材的架构演进顺序，历史实验不代表这些网络在今天的性能排名。

## 核心关系与公式

对输入高度 $H$，核大小 $K$、每侧填充 $P$、步幅 $S$、膨胀率 $D$ 的普通卷积，输出高度为

$$H_{out}=\left\lfloor\frac{H+2P-D(K-1)-1}{S}+1\right\rfloor.$$

宽度同理。忽略偏置，普通卷积参数量为 $K_hK_wC_{in}C_{out}$；分成 $g$ 组时为其 $1/g$，前提是通道数满足分组要求。$1\times1$ 卷积主要混合通道；步幅为 1 时不改变空间分辨率。

批量规范化（batch normalization，BN）在指定统计轴上计算均值 $\mu_B$ 与方差 $\sigma_B^2$：

$$y=\gamma\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}+\beta.$$

$\gamma,\beta$ 是可学习缩放与平移，$\epsilon>0$ 防止数值不稳。卷积 BN 通常对每个通道，沿批量和空间轴统计；预测通常使用训练期间累计的运行统计量。层规范化（layer normalization）按单个样本的指定特征轴统计，两者不能仅凭公式相似就互换。

## 按教材顺序解析

### 8.1 深度卷积网络 AlexNet

[原文与代码](../../chapter_convolutional-modern/alexnet.md)。核心转变是由手工特征转向表示学习（representation learning），其成功也依赖数据与计算资源。ReLU 改变激活与梯度性质，暂退法（dropout）用于控制容量；末端大规模全连接层是理解参数开销的重点。教材使用缩放后的 Fashion-MNIST 展示训练流程，不能把该示例等同于原始 ImageNet 实验。

### 8.2 使用块的网络 VGG

[原文与代码](../../chapter_convolutional-modern/vgg.md)。把多层卷积与池化封装为块（block）后，网络可以由“每块层数与通道数”的配置生成。连续小卷积核既扩大有效感受野（receptive field），又在层间增加非线性；即使感受野相同，也不等价于一个大卷积核。注意池化使空间尺寸下降，通道数通常逐阶段增加。

### 8.3 网络中的网络 NiN

[原文与代码](../../chapter_convolutional-modern/nin.md)。NiN 在局部空间位置使用 $1\times1$ 卷积及非线性，增强通道间的表达能力，再以全局平均池化（global average pooling）替代巨大的全连接头。最后的通道数对应类别数，空间维度取平均后得到类别分数；这不是对所有通道一起求平均。

### 8.4 多分支网络 GoogLeNet

[原文与代码](../../chapter_convolutional-modern/googlenet.md)。Inception 块同时处理多种卷积核与池化分支，先以 $1\times1$ 卷积压缩部分分支的通道数，再进行较昂贵的空间卷积。分支输出沿通道轴拼接，要求其高度和宽度一致；输出通道数是各分支通道数之和。由此建立输入干线、主体、预测头的分工思路。

### 8.5 批量规范化

[原文与代码](../../chapter_convolutional-modern/batch-norm.md)。先理解统计轴，再区分训练与预测的统计来源，最后看可学习参数如何保留表达能力。BN 常能改善训练，但“减少内部协变量偏移就解释了全部收益”不是教材认可的定论。小批量、分布变化和训练模式误用都会影响行为；教材从零实现是理解机制的教学代码，工程模型应明确使用框架的训练/评估模式。

### 8.6 残差网络 ResNet 与 ResNeXt

[原文与代码](../../chapter_convolutional-modern/resnet.md)。残差连接（residual connection）用 $y=x+F(x)$ 学习相对输入的修正；若残差分支趋于零，具有形状一致的捷径便可传递输入。源码块还可能在相加后使用 ReLU，因此应区分“加法处的恒等捷径”和整个块对任意输入的恒等映射。降采样或变通道时需要投影捷径。ResNeXt 引入分组卷积（grouped convolution），通过分组数等变量在表达能力和成本之间取舍。

### 8.7 稠密连接网络 DenseNet

[原文与代码](../../chapter_convolutional-modern/densenet.md)。DenseNet 将历史特征拼接，而 ResNet 将同形状特征相加。若初始通道为 $c_0$，块内每层新增 $k$ 个通道，经过 $L$ 层后通道为 $c_0+Lk$；$k$ 是增长率（growth rate）。过渡层（transition layer）压缩通道并降低分辨率。特征复用不等于低显存占用，拼接与保存中间激活可能带来明显开销。

### 8.8 卷积网络架构设计

[原文与代码](../../chapter_convolutional-modern/cnn-design.md)。AnyNet 将深度、宽度、瓶颈比例和分组等整理为设计空间（design space），RegNet 再根据系统实验约束这个空间。应学习如何比较一个设计空间中模型的性能分布，而不只挑选单次最优模型。公平比较必须控制训练配置、预算和评估协议；教材结论有其数据与计算条件。

## 代码阅读路线

依次追踪 [VGG 的 `vgg_block`](../../chapter_convolutional-modern/vgg.md)、[NiN 的 `nin_block`](../../chapter_convolutional-modern/nin.md)、[GoogLeNet 的 `Inception`](../../chapter_convolutional-modern/googlenet.md)，手写每个块前后的 `(批量, 通道, 高, 宽)`。再读 [BN 的 `batch_norm` 与 `BatchNorm`](../../chapter_convolutional-modern/batch-norm.md)，检查统计轴及运行状态；最后对照 [ResNet 的 `Residual`、`ResNeXtBlock`](../../chapter_convolutional-modern/resnet.md) 与 [DenseNet 的 `DenseBlock`、`transition_block`](../../chapter_convolutional-modern/densenet.md)。先完成单块前向追踪，再读整网训练。

## 易错点与自测

- **参数少是否就运行快？** 不一定。答案线索：中间激活、分支调度、内存访问以及硬件执行效率也决定耗时。
- **两个分支都输出 32 个通道就能相加吗？** 还需批量与空间形状一致；不能依靠意外广播掩盖结构错误。
- **BN 的运行均值是否由反向传播直接学习？** 通常不是。区分统计更新与 $\gamma,\beta$ 的梯度更新。
- **DenseNet 与 ResNet 为什么不能只替换加号？** 拼接改变通道数，后续卷积的输入维度与内存需求随之改变。

## 小实验建议（本次未执行）

1. 固定输入、训练轮数和随机种子，比较一个全连接分类头与全局平均池化头。记录参数量、训练/验证指标和峰值显存；若性能不同，不能单凭一次运行归因于参数数量。
2. 在同形状残差块中分别观察训练模式和评估模式下 BN 的输出；固定参数，改变一个样本所在批次的其他样本。该实验检验批量依赖性，不用于证明 BN 收益的完整理论解释。
