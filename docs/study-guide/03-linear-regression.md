# 第 3 章：线性神经网络回归——完整训练系统的最小实例

[返回导读目录](README.md) · [教材原章](../../chapter_linear-regression/index.md) · [重点术语表](../GLOSSARY.zh-CN.md)

本章用线性回归（linear regression）串起数据、模型、损失、优化与评估。需要第 2 章的矩阵乘法和自动微分；后续分类与深层网络主要替换模型和损失，继续复用这里的训练骨架。学完应能定位一次参数更新的每个步骤，并解释为什么训练误差小还不够。

## 按教材目录逐节解析

### 3.1 线性回归（Linear Regression）

[源文：linear-regression.md](../../chapter_linear-regression/linear-regression.md)。从房屋面积、房龄到价格，建立带偏置的仿射模型；平方损失让预测残差可求导。解析解、小批量随机梯度下降和向量化是三条相关线索：解析解帮助理解目标，梯度法适合复用到复杂模型，向量化把样本级循环变成高效张量运算。高斯噪声假设解释了平方损失与极大似然估计（maximum likelihood estimation）的联系。

### 3.2 面向对象设计（Object-Oriented Design for Implementation）

[源文：oo-design.md](../../chapter_linear-regression/oo-design.md)。`Module` 负责模型、训练步与优化器配置，`DataModule` 负责批量数据，`Trainer` 组织多轮训练。`add_to_class` 允许教材分段补充类方法，`HyperParameters` 与 `ProgressBoard` 减少重复记录和绘图。这里有教学用接口骨架，完整行为会在后文逐步加入；单独复制一段类定义不等于拥有完整训练器。

### 3.3 合成回归数据（Synthetic Regression Data）

[源文：synthetic-regression-data.md](../../chapter_linear-regression/synthetic-regression-data.md)。`SyntheticRegressionData` 先指定真实权重和偏置，再生成输入与噪声，因此可以同时检查预测误差和参数估计误差。数据迭代器先手写索引和打乱，再用 `TensorDataset`、`DataLoader` 简化；训练与验证切片必须分开，不能因数据是合成的就重复使用同一批样本评价泛化。

### 3.4 从零实现（Linear Regression Implementation from Scratch）

[源文：linear-regression-scratch.md](../../chapter_linear-regression/linear-regression-scratch.md)。`LinearRegressionScratch` 直接持有 `w`、`b`，前向计算为矩阵乘法加偏置，损失先取均值。`SGD` 与 `fit_epoch` 连接清梯度、反向传播、更新参数和验证。确认标签与预测均为 `(batch,1)`；否则广播可能让程序运行却优化错误目标。

### 3.5 简洁实现（Concise Implementation of Linear Regression）

[源文：linear-regression-concise.md](../../chapter_linear-regression/linear-regression-concise.md)。用 `nn.LazyLinear(1)`、`nn.MSELoss` 与框架优化器替代手写组件，训练流程保持相同。PyTorch 的 `MSELoss` 没有从零版的 $1/2$ 因子，所以梯度尺度不同；同一学习率下不能仅凭曲线差异判断两份实现不等价。延后确定输入维度的机制在第 6 章继续解释。

### 3.6 泛化（Generalization）

[源文：generalization.md](../../chapter_linear-regression/generalization.md)。区分训练误差、总体泛化误差和有限验证集给出的估计。欠拟合（underfitting）与过拟合（overfitting）是诊断线索，不是只看一个差值就能确定的病因；训练不足也可能让训练误差高。验证集（validation set）用于选择模型与超参数，测试集（test set）用于最终独立评估，$K$ 折交叉验证（cross-validation）重复的是划分与拟合流程。

### 3.7 权重衰减（Weight Decay）

[源文：weight-decay.md](../../chapter_linear-regression/weight-decay.md)。用少量样本、高维特征放大过拟合问题，比较加入平方范数惩罚前后的行为。从零版在损失中加惩罚，简洁版在 SGD 参数组中指定 `weight_decay`，并让偏置不衰减。权重衰减（weight decay）与 L2 正则化在本章 SGD 更新下对应；不要无条件推广到所有优化器。

## 关键公式与条件

对 $n$ 个样本、$d$ 个输入特征，令 $X\in\mathbb R^{n\times d}$、$w\in\mathbb R^{d\times1}$、$b$ 为标量：

$$
\hat y=Xw+b,\qquad
L(w,b)=\frac1{2n}\|Xw+b-y\|_2^2.
$$

平方惩罚会更重视大残差。若 $y_i=w^\top x_i+b+\epsilon_i$，且噪声独立、服从固定方差的零均值高斯分布（Gaussian distribution），最大化条件似然等价于最小化平方损失。离开这些假设，平方损失仍可使用，但上述概率解释不再自动成立。

对小批量 $\mathcal B$，学习率为 $\eta$，正则化系数为 $\lambda$：

$$
w\leftarrow(1-\eta\lambda)w
-\frac{\eta}{|\mathcal B|}\sum_{i\in\mathcal B}
 x_i(\hat y_i-y_i).
$$

前一项缩小权重，后一项拟合数据。若损失已经取均值，更新时不应再除一次批量大小。偏置可以有独立更新规则。把偏置吸收进常数特征后，正规方程可写为 $(X^\top X)^{-1}X^\top y$，但显式逆要求相应矩阵可逆；存在共线性时不能直接套用这个形式。

## 代码阅读路线

先读 `oo-design.md` 的三个核心类，再读 `synthetic-regression-data.md` 的 `get_dataloader`，接着沿 `training_step → loss → backward → SGD.step` 追踪 `linear-regression-scratch.md`。确认真实参数 `[2, -3.4]` 和偏置 `4.2` 来自合成过程，而非模型预先知道的答案。最后比较 `linear-regression-concise.md` 与 `weight-decay.md` 的参数组，理解哪些细节交给了框架。

应同时看训练损失、验证损失和参数误差；含噪有限样本中，参数不精确等于生成值并不意味着训练失败。不同特征尺度也会改变优化速度与惩罚含义，因此不能只靠无限增加训练轮数解决所有问题。

## 自测与回答线索

1. 为什么本章先用已知真实参数的合成数据？线索：既能测预测，也能查参数恢复和训练流程。
2. 从零版损失已平均，SGD 再除批量大小会怎样？线索：有效步长额外缩小。
3. 权重衰减为什么会改变最优参数？线索：目标新增了惩罚，最优点不只由数据残差决定。
4. 验证误差远高于训练误差时应检查什么？线索：容量、样本量、正则化、划分是否一致，以及预处理或分布差异。

## 小实验建议（未执行）

- 固定数据和初始化，分别用半均方误差与均方误差训练。相应调整学习率后比较参数轨迹，记录损失定义，避免把尺度变化误当算法优势。
- 复用 `weight-decay.md` 的 200 维、20 个训练样本设置，比较数个 $\lambda$ 的训练误差、验证误差与权重范数。保留相同数据划分并重复随机种子；预期权衡只是待验证假设，本文没有生成训练结果。
