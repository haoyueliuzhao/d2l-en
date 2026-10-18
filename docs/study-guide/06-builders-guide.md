# 第 6 章：深度学习构建指南——掌握模型的生命周期

[返回导读目录](README.md) · [教材原章](../../chapter_builders-guide/index.md) · [重点术语表](../GLOSSARY.zh-CN.md)

本章不引入新的预测任务，重点是把“能训练一个例子”提升为“能组合、检查、保存和迁移一个模型”。先修为第 3–5 章的训练与 MLP；后续复杂网络依赖这里的模块、参数管理、自定义层和 GPU 设备规则。建议始终选择 PyTorch 分支连贯阅读，避免混用其他框架的对象生命周期。

## 按教材目录逐节解析

### 6.1 层与模块（Layers and Modules）

[源文：model-construction.md](../../chapter_builders-guide/model-construction.md)。模块既可以是一层，也可以包含子模块、参数和控制流。先看自定义 `MLP` 如何定义 `forward`，再看 `MySequential` 如何注册并依次调用子层，最后读 `FixedHiddenMLP` 与 `NestMLP`。关键不是“所有前向计算都要写成顺序表”，而是把参与状态管理的组件注册到模型中，保证参数能被发现和保存。

### 6.2 参数管理（Parameter Management）

[源文：parameters.md](../../chapter_builders-guide/parameters.md)。区分参数对象、其中的数值与梯度；梯度尚未计算时可能是 `None`。通过层索引、命名参数和状态字典检查形状与内容。参数共享（parameter sharing）意味着多个计算位置使用同一个参数对象，不是两个张量碰巧初始化成相同数值；共享参数获得来自多条路径的梯度贡献。

### 6.3 参数初始化（Parameter Initialization）

[源文：init-param.md](../../chapter_builders-guide/init-param.md)。用正态、常数、Xavier 与自定义规则说明怎样按层类型设置参数，`apply` 将函数递归应用到模块。初始化不是训练更新；在已训练模型上再次运行初始化会改变已学参数。应先确认要初始化的层、权重和偏置范围，不要因为一个检查函数顺手调用了初始化而丢失结果。

### 6.4 延后初始化（Lazy Initialization）

[源文：lazy-init.md](../../chapter_builders-guide/lazy-init.md)。`LazyLinear` 在拿到输入前尚不确定权重的输入维度；第一次前向计算让形状落实。`apply_init` 先传入样例输入，再运行初始化器。这样修改上游结构时更方便，但排错、打印参数形状或加载权重前，应知道模型是否已完成形状推断，不能把未物化参数当普通张量处理。

### 6.5 自定义层（Custom Layers）

[源文：custom-layer.md](../../chapter_builders-guide/custom-layer.md)。`CenteredLayer` 演示无参数层，`MyLinear` 演示通过 `nn.Parameter` 注册可训练权重。前者执行 `X-X.mean()`，是对整个输入求均值，不等同于按特征的批量规范化（batch normalization）。当前 PyTorch 源码的 `MyLinear.forward` 使用 `self.weight.data` 和 `self.bias.data`；这种写法会绕开这些参数的梯度跟踪。学习可训练层时应直接使用参数对象，如 `X @ self.weight + self.bias`，并理解“注册参数”和“实际让参数参与计算图”是两件事。此处是静态阅读提示，教材源文件未修改。

### 6.6 文件读写（File I/O）

[源文：read-write.md](../../chapter_builders-guide/read-write.md)。先保存张量、列表和字典，再保存模型的 `state_dict` 并装入相同结构的新实例。状态字典保存权重等状态，不替你保存网络结构的 Python 定义。若目的是继续训练，仅保存模型权重通常不够，还需优化器状态、训练位置及必要的随机状态；这是对本节恢复场景的延伸，不是该示例已经实现的功能。

### 6.7 GPU（GPUs）

[源文：use-gpu.md](../../chapter_builders-guide/use-gpu.md)。`try_gpu` 和 `try_all_gpus` 查询可用设备；张量与模型参数需要放到一致设备才能参与相应运算。查看 `Trainer.prepare_batch` 与 `prepare_model` 如何连接数据迁移和训练。频繁从 GPU 取标量回 CPU 可能触发同步，影响吞吐；“已经放到 GPU”不保证小任务一定更快，多 GPU 的训练策略也不由 `.to(device)` 自动完成。

## 用三个关系理解模块行为

对输入 $X\in\mathbb R^{n\times d}$，自定义线性层可以写成

$$
Y=\operatorname{ReLU}(XW+b),\qquad W\in\mathbb R^{d\times h},\quad b\in\mathbb R^h.
$$

参数必须保留从 $Y$ 回到 $W,b$ 的可微依赖；数值上得到正确输出，不足以证明能训练。源码自定义 `MyLinear` 使用 `(in_units, units)` 权重排列，而内置 `nn.Linear` 的权重排列不同，手工复制权重时要核对转置。

若同一个参数 $W$ 在图中被使用两次，损失分别经过两条路径依赖它，则

$$
\nabla_W L=\left.\nabla_WL\right|_{\text{路径 1}}
+\left.\nabla_WL\right|_{\text{路径 2}}.
$$

共享减少独立参数数目，但不意味着只保留某一次使用的梯度，也不意味着计算量按相同比例减少。

在结构相同、参数与必要状态一致、同为评估模式且不存在额外随机操作时，保存再加载的模型应对相同输入给出数值一致或在相应精度容差内一致的输出。该条件说明文件读写应该验证什么，而不是本文已经执行的恢复结果。

## 代码阅读路线与边界

按 `MLP → MySequential → parameters.md 的共享层 → apply_init → MyLinear → state_dict → prepare_batch` 阅读。每一步只问四个问题：对象注册在哪里、数值位于什么设备、梯度从哪里来、保存时包含什么。暂时不需要重写整套训练框架。

尤其区分 `model.eval()`、禁用梯度和冻结参数：评估模式控制 dropout 等层的行为，禁用梯度控制记录计算，冻结参数控制相应变量是否需要梯度，三者各自处理不同问题。含随机操作的前向传播不能无条件用于恢复一致性比较。

## 自测与回答线索

1. 为什么参数数值相同不一定是共享参数？线索：修改一个对象，另一个是否指向同一存储和参数实例。
2. 为什么 `nn.Parameter` 出现在 `parameters()` 中仍可能没有梯度？线索：它可能没参与损失路径，或通过 `.data`、`detach` 截断了路径。
3. 为什么只加载 `state_dict` 还需要创建模型？线索：状态字典不包含完整网络执行逻辑。
4. 将输入移到 GPU 后为什么仍可能报设备错误？线索：模型参数或其他参与计算的张量还在 CPU。

## 小实验建议（未执行）

- 在自己的练习文件中比较原示例 `MyLinear` 与直接使用 `Parameter` 的版本，查看输出的梯度属性及权重 `grad`。将可能无法反向传播的情况记为观察结果，不要把它包装成模型训练成功。
- 创建小型 MLP，先物化延后初始化参数，再保存并加载到新实例，比较评估模式输出；随后有意改变结构，解释加载报错与形状不匹配的关系。无需完整训练或占用多块 GPU。
