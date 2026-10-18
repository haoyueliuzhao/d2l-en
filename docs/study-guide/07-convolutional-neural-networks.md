# 第 7 章：卷积神经网络——把图像结构变成模型约束

[返回导读目录](README.md) · [教材原章](../../chapter_convolutional-neural-networks/index.md) · [重点术语表](../GLOSSARY.zh-CN.md)

本章解释为何图像不宜只当成无序特征向量，并逐步搭建 LeNet。先修是矩阵与张量、MLP、分类损失和模块构建；后续现代 CNN、目标检测与分割都依赖这里的卷积、通道、步幅和感受野。学习目标是能手算一个输出元素、每层形状与参数量，而不只是会调用 `Conv2d`。

## 按教材目录逐节解析

### 7.1 从全连接层到卷积（From Fully Connected Layers to Convolutions）

[源文：why-conv.md](../../chapter_convolutional-neural-networks/why-conv.md)。从具有空间索引的全连接映射出发，加入局部性与位置共享的约束，得到卷积结构。局部连接限制每个输出关注的输入范围，参数共享（parameter sharing）让同一特征检测器在不同位置复用。原文讨论的平移性质要区分：卷积特征通常追求平移等变性（translation equivariance），图像类别预测可能追求平移不变性（translation invariance）；前者是输出随输入移动，后者是输出不变。

### 7.2 图像卷积（Convolutions for Images）

[源文：conv-layer.md](../../chapter_convolutional-neural-networks/conv-layer.md)。手写 `corr2d`，用 `[1,-1]` 检测像素变化，再从输入和目标输出学习卷积核。多数深度学习框架实际执行互相关（cross-correlation），没有数学卷积中的核翻转；学习自由卷积核时，两者只是参数表示不同。特征图（feature map）保留空间位置，感受野（receptive field）描述某个输出可能依赖的原始输入区域。

### 7.3 填充与步幅（Padding and Stride）

[源文：padding-and-strides.md](../../chapter_convolutional-neural-networks/padding-and-strides.md)。填充（padding）控制边界和尺寸收缩，步幅（stride）控制窗口移动距离及下采样。阅读时分清教材公式中的两侧总填充与框架参数中的每侧填充；不同含义不能直接代入同一个式子。填充改变边界附近的输入内容，步幅大于 1 会丢弃部分空间采样，并影响精确的平移对应关系。

### 7.4 多输入与多输出通道（Multiple Input and Multiple Output Channels）

[源文：channels.md](../../chapter_convolutional-neural-networks/channels.md)。一个输出通道对所有输入通道分别相关再求和；多个输出通道使用多组卷积核。`1×1` 卷积不看相邻像素，却能在每个空间位置混合通道、改变通道数，因此不是无意义的恒等操作。通道代表可学习特征，不必一直对应 RGB 颜色。

### 7.5 汇聚（Pooling）

[源文：pooling.md](../../chapter_convolutional-neural-networks/pooling.md)。最大汇聚（max pooling）取局部最大值，平均汇聚（average pooling）取局部均值，两者没有需要学习的卷积核。它们分别处理各输入通道，不在通道间求和。汇聚可使表示对某些局部变化更稳健，但不能据此声称任何平移都完全不改变网络输出。

### 7.6 LeNet（Convolutional Neural Networks (LeNet)）

[源文：lenet.md](../../chapter_convolutional-neural-networks/lenet.md)。把两个卷积块与三层全连接组合成完整分类器，复用 `Classifier`、Fashion-MNIST 与训练器。本仓库实现采用 sigmoid 与平均汇聚；理解它的具体设计后，再在下一章比较新架构。`layer_summary` 用真实前向传播逐层展示形状，`apply_init` 使延后初始化层先获得确定尺寸。

## 三个必须掌握的计算

忽略批量轴，对无扩张的多通道互相关：

$$
Y_{o,i,j}=b_o+\sum_{c=1}^{C_{\rm in}}\sum_{u=0}^{K_h-1}
\sum_{v=0}^{K_w-1}K_{o,c,u,v}X_{c,i+u,j+v}.
$$

$o$ 为输出通道，$c$ 为输入通道，$(i,j)$ 为输出位置，$(u,v)$ 为核内位置。这里写的是步幅 1 且已处理边界的情形。普通卷积含偏置时的参数量为

$$
C_{\rm out}(C_{\rm in}K_hK_w+1).
$$

它不随输入空间尺寸增加，但同一参数会在很多位置参与计算，因此参数少不代表运算一定少。

令输入高为 $H$，核高 $K_h$，两侧对称的每侧填充为 $P_h$，步幅为 $S_h$，且扩张率为 1，则

$$
H_{\rm out}=\left\lfloor\frac{H+2P_h-K_h}{S_h}\right\rfloor+1.
$$

宽度同理。这一写法明确采用“每侧填充”；若使用教材中的总填充变量，就应把 $2P_h$ 替换为总量。

对连续空间层，可用辅助递推理解感受野：$r_0=j_0=1$，$r_l=r_{l-1}+(k_l-1)j_{l-1}$，$j_l=j_{l-1}s_l$。其中 $r_l$ 为理论感受野宽度，$j_l$ 为该层相邻位置在原输入上的间隔，$k_l,s_l$ 为核宽和步幅；此式假定扩张率为 1。理论覆盖不等于各像素对预测的实际影响都相同。

## 沿 LeNet 核对形状

对本章 PyTorch 实现输入 `(N,1,28,28)`，按照源码得到下列形状。这是结构推算，不是新增训练结果。

| 运算 | 输出形状 | 要点 |
| --- | --- | --- |
| `5×5` 卷积，6 通道，每侧填充 2 | `(N,6,28,28)` | sigmoid 不改变形状 |
| `2×2` 平均汇聚，步幅 2 | `(N,6,14,14)` | 通道数保持 6 |
| `5×5` 卷积，16 通道，无填充 | `(N,16,10,10)` | 每个输出通道融合 6 个输入通道 |
| `2×2` 平均汇聚，步幅 2 | `(N,16,5,5)` | 展平后有 400 个特征 |
| 全连接 120、84、10 | `(N,10)` | 最后一层输出类别分数 |

## 代码路线与自测

先在 `conv-layer.md` 手算 `corr2d` 的一个输出，再用 `padding-and-strides.md` 的 `comp_conv2d` 核对尺寸，然后读 `channels.md` 的 `corr2d_multi_in_out` 与 `corr2d_multi_in_out_1x1`，最后跟踪 `LeNet.layer_summary`。PyTorch 使用 `(N,C,H,W)`，不要照搬通道放在最后的其他框架布局。

1. 三输入通道、五输出通道、`3×3` 卷积有多少参数？线索：含偏置为 $5(3\cdot3\cdot3+1)=140$。
2. `1×1` 卷积为什么有表达作用？线索：每个位置都进行可学习的通道线性组合。
3. 汇聚为什么不会自动合并 RGB 通道？线索：各通道独立执行空间汇总。
4. 连续两个步幅 1 的 `3×3` 层有多大理论感受野？线索：单个维度从 1 增至 3，再增至 5；边界填充值需另行解释。

## 小实验建议（未执行）

- 复用竖直边缘示例，对图像及其转置分别应用 `[1,-1]` 和其转置，画出输出并解释方向选择性；再用小批量更新学习核，记录核值和误差，而非只看图像。
- 固定数据划分和训练预算，比较 MLP 与 LeNet 的参数量、训练时间和验证指标。先核对输入预处理一致，计时时明确是否包含数据加载与 GPU 同步；本文未执行该比较，也不预设哪个指标一定更好。
