# 第 13 章：计算性能重点解析

[学习导览](README.md) · [本章原文](../../chapter_computational-performance/index.md) · [术语对照表](../GLOSSARY.zh-CN.md)

本章把“算得对”推进到“知道时间花在哪里”。先修是训练循环、设备与张量、梯度聚合。目标是区分计算、数据移动、同步和 Python 调度的成本，设计能够回答问题的性能测量。教材中的硬件型号、延迟数字和框架接口属于其示例环境；这里保留其教学逻辑，不把旧测量值当作当前服务器实测。

## 核心关系与公式

数据并行（data parallelism）中，第 $k$ 个设备处理 $B_k$ 个样本，得到局部平均梯度 $\bar g_k$，全局平均应为

$$\bar g=\sum_{k=1}^{K}\frac{B_k}{\sum_jB_j}\bar g_k.$$

只有各设备批量相同时，才能直接对局部均值等权平均。若先求局部梯度和，则应聚合全部和后仅除以一次全局样本数。误把求和、平均或归一化重复执行，会悄悄改变有效学习率。

并行加速比（speedup）为 $S_K=T_1/T_K$，并行效率为 $E_K=S_K/K$；比较时须说明固定的是总样本工作量、全局批量还是每卡批量。理想化的 Amdahl 定律写为

$$S_K\le\frac1{(1-p)+p/K},$$

$p$ 为可并行工作比例，忽略通信等额外成本。它说明串行部分限制加速上界，不是具体训练任务的精确性能预测。

单次传输可粗略写成 $T\approx\text{延迟}+\text{字节数}/\text{带宽}$；因此合并许多小传输常能减少启动开销。实际并发与拥塞会使估算偏离测量。

## 按教材顺序解析

### 13.1 编译器与解释器

[原文与代码](../../chapter_computational-performance/hybridize.md)。命令式编程（imperative programming）便于调试和动态控制，符号式编程（symbolic programming）通过先构建计算结构再执行提供优化机会。教材展示混合编程和模型序列化，PyTorch 示例采用 `torch.jit.script`。阅读重点是捕获计算、优化与首次编译开销的关系；不能只计第一次调用就判断稳定运行速度，也不能假定任意 Python 控制流都能被同样处理。

### 13.2 异步计算

[原文与代码](../../chapter_computational-performance/async-computation.md)。前端提交操作后，GPU 后端可能尚未完成；CPU 计时器因此可能测到入队时间而非执行时间。公平测量应先预热，在计时开始前清空相关待执行工作，在计时结束前等待被测工作完成。读取 GPU 标量到 CPU、复制结果等动作可能造成同步；训练中每步增加这些动作也会改变被测程序本身。

### 13.3 自动并行

[原文与代码](../../chapter_computational-performance/auto-parallelism.md)。没有数据依赖的不同设备任务，以及某些计算与通信，可以重叠执行。并发能否产生收益取决于依赖、调度、传输方式和硬件资源；把两段代码相邻写下并不能证明它们重叠。结束测量时需要等待所有涉及设备或流完成，否则快的只是提交过程。

### 13.4 硬件

[原文](../../chapter_computational-performance/hardware.md)。寄存器、缓存、内存、存储与网络具有不同容量、延迟和带宽，数据复用常比增加算术速度更重要。向量化（vectorization）与大块连续访问有助于提高设备利用率。低精度可减少传输与计算成本，也带来数值范围和误差问题；不能把某一代芯片对某精度的优势泛化到所有设备。先估算数据量和算术量，再用性能分析器（profiler）定位主要瓶颈。

### 13.5 多 GPU 训练

[原文与代码](../../chapter_computational-performance/multiple-gpus.md)。教材先复制参数、拆分批量，各卡独立前向和反向，然后聚合梯度并更新各副本。所有副本应从相同状态出发，执行一致更新；否则它们会逐渐分叉。教学用 `allreduce` 展示归约（reduction）和广播（broadcast）的逻辑，其 Python 层实现不能代表高性能通信库的速度。

### 13.6 多 GPU 的简洁实现

[原文与代码](../../chapter_computational-performance/multiple-gpus-concise.md)。框架封装数据分发、模型副本和梯度聚合，本节 PyTorch 使用 `nn.DataParallel`。理解封装前后哪些责任仍由用户承担：设备选择、输入放置、初始化、全局批量与训练模式。复杂模型可能更能摊薄通信开销，但添加 GPU 后不一定加速，尤其在小模型、小批量或数据加载受限时。

### 13.7 参数服务器

[原文](../../chapter_computational-performance/parameterserver.md)。参数服务器（parameter server）集中或分片管理参数更新，环形同步（ring synchronization）则在设备之间分阶段归约与分发数据。关键比较对象是通信拓扑、链路带宽、延迟和聚合瓶颈，而不只是 GPU 数。多机情境还要考虑慢工作节点、跨机链路和一致性策略；同步训练等待各方，异步更新可能读取较旧参数，二者的优化行为也可能不同。

## 代码阅读路线与计时口径

先读 [异步计算的预热与 `torch.cuda.synchronize`](../../chapter_computational-performance/async-computation.md)，再跟踪 [自动并行的 `run`、`copy_to_cpu`](../../chapter_computational-performance/auto-parallelism.md)。随后逐行追踪 [多 GPU 的 `get_params` → `split_batch` → `train_batch` → `allreduce`](../../chapter_computational-performance/multiple-gpus.md)：本版各卡先对损失求和，聚合后更新函数按全局批量归一化。最后看 [简洁实现的 `train`](../../chapter_computational-performance/multiple-gpus-concise.md)，理解框架替代了哪些步骤。

若自行测量，应至少记录设备、框架版本、精度、输入形状、批量、预热次数、重复次数及是否含数据加载/传输。用于隔离算子执行的同步计时和用于评估真实流水线吞吐的计时回答不同问题：后者不应无理由在每个算子后同步，从而破坏重叠。可报告中位数与波动范围，避免只挑最快一次。

## 易错点与自测

- **GPU 操作后 Python 很快返回，说明计算很快吗？** 不能据此判断；先确认执行完成及计时边界。
- **两张卡各自算出平均梯度，直接相加是否正确？** 若不再归一化就会改变尺度；还须处理各卡样本数不等的情况。
- **多卡训练更快且准确率更高，是否证明并行算法更好？** 先核对全局批量、学习率、样本数和更新次数是否改变。
- **减少 CPU 使用率能否提高 GPU 吞吐？** 不必然；数据解码和预处理可能需要 CPU 并行，关键是定位阻塞环节。

## 小实验建议（本次未执行）

1. 对同一个 GPU 矩阵乘法分别测量“仅提交”“预热后同步完成”的时间，说明二者的含义。重复多次，避免把初次分配和内核初始化误当成稳定计算开销。
2. 若有至少两张 GPU，分别固定全局批量和固定每卡批量比较吞吐、每步时间和验证指标，明确两种实验的工作量不同。若只有一张 GPU，保留单卡基线即可，不生成虚构多卡结果。
