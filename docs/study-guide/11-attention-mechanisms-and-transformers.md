# 第 11 章：注意力机制与 Transformer 重点解析

[学习导览](README.md) · [本章原文](../../chapter_attention-mechanisms-and-transformers/index.md) · [术语对照表](../GLOSSARY.zh-CN.md)

本章从“根据当前需求加权取信息”推导注意力，再将其组成 Transformer。先修是矩阵乘法、softmax、序列到序列模型、残差连接和规范化。学习目标是画出查询、键、值的来源与维度，解释掩码如何防止无效信息进入计算，并能区分文本和视觉模型的结构选择。

## 核心关系与公式

缩放点积注意力（scaled dot-product attention）为

$$A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right),\qquad O=AV.$$

省略批量维度时，$Q\in\mathbb R^{n_q\times d_k}$，$K\in\mathbb R^{n_k\times d_k}$，$V\in\mathbb R^{n_k\times d_v}$；注意力权重 $A$ 为 $n_q\times n_k$，输出为 $n_q\times d_v$。softmax 沿键的位置轴归一化。掩码 $M$ 在允许位置为 0，禁止位置取负无穷或足够大的负数；后者是数值近似。

$1/\sqrt{d_k}$ 缓和点积分数随维度增大而增大的尺度；常见方差解释依赖分量近似独立、零均值和单位方差等假设。若某一查询没有任何合法键，应单独定义处理方式：整行有限负常数可能得到均匀分布，整行负无穷可能得到非有限值。

多头注意力（multi-head attention）分别投影到多个子空间，拼接各头输出后再投影。教材等宽分头实现要求隐藏维度能被头数整除。以序列长度 $n$、表示维度 $d$ 计，标准全量自注意力的两次主要矩阵乘法约为 $O(n^2d)$，注意力矩阵存储为 $O(n^2)$；完整模块还包含线性投影等成本。

## 按教材顺序解析

### 11.1 查询、键和值

[原文与代码](../../chapter_attention-mechanisms-and-transformers/queries-keys-values.md)。查询（query）代表当前检索需求，键（key）用来计算匹配程度，值（value）是被加权汇总的信息。三者不是固定语义标签，而是模型中三种计算角色。注意力热力图展示权重分配，但高权重本身不构成因果解释。

### 11.2 注意力汇聚

[原文与代码](../../chapter_attention-mechanisms-and-transformers/attention-pooling.md)。Nadaraya–Watson 核回归（kernel regression）按查询与观测位置的距离对观测值加权，建立注意力的直观起点。核的形状与宽度影响局部性和平滑程度：较窄时可能更贴近邻近样本，也可能更敏感于噪声。权重取决于距离及观测分布，不只是取最近的一个点。

### 11.3 注意力评分函数

[原文与代码](../../chapter_attention-mechanisms-and-transformers/attention-scoring-functions.md)。点积要求参与点积的查询与键维度一致；加性注意力（additive attention）先映射两者再打分，可处理原始维度不同的情况。`masked_softmax` 接受每个样本或每个查询的有效长度，先屏蔽分数再归一化。批量矩阵乘法的批量维必须对齐，不能把广播碰巧成功当成语义正确。

### 11.4 Bahdanau 注意力

[原文与代码](../../chapter_attention-mechanisms-and-transformers/bahdanau-attention.md)。解码器上一步隐藏状态作为查询，编码器所有时间步的隐藏状态作为键和值，为当前输出重新形成上下文向量。这样缓解“整句挤进一个最终状态”的信息瓶颈，但编码和解码仍包含 RNN 的时间递推。源句填充位置必须通过有效长度排除。

### 11.5 多头注意力

[原文与代码](../../chapter_attention-mechanisms-and-transformers/multihead-attention.md)。不同头使用不同投影，允许模型在多个表示子空间汇聚信息。代码通过 reshape、transpose 将头维暂时并入批量维实现并行，之后必须按逆过程恢复。增加头数时若总隐藏维度固定，每头维度会变小；不能直接把头数当成模型容量的单向度指标。

### 11.6 自注意力与位置编码

[原文与代码](../../chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.md)。自注意力（self-attention）的 Q、K、V 来自同一序列，跨注意力（cross-attention）则可来自不同序列。没有位置或其他顺序信息时，自注意力对输入排列具有相应的置换等变性，因此需要位置编码（positional encoding）。教材的正弦与余弦编码提供绝对位置表示，也可用线性关系表达位移结构；这不等于任意长序列上必然泛化。

### 11.7 Transformer

[原文与代码](../../chapter_attention-mechanisms-and-transformers/transformer.md)。编码器块包含自注意力、逐位置前馈网络（positionwise feed-forward network）与残差规范化；前馈网络共享参数，但分别处理各位置。解码器另外包含因果自注意力与连接源表示的跨注意力。填充掩码与因果掩码目的不同：前者过滤无效 token，后者禁止访问未来目标。教材 `AddNorm` 为加法后规范化；训练并行处理目标序列，逐 token 推断使用缓存，需逐层保持历史一致。

### 11.8 视觉 Transformer

[原文与代码](../../chapter_attention-mechanisms-and-transformers/vision-transformer.md)。视觉 Transformer（vision Transformer，ViT）将图像切成图像块（patch），投影为 token，再加入分类 token 与位置表示。若高宽均可被块边长 $P$ 整除，图像 token 数为 $(H/P)(W/P)$，加分类 token 后再多 1。减小块大小或提高分辨率会迅速增加注意力成本。本节使用可学习位置嵌入，`ViTBlock` 的规范化顺序也与上一节 `AddNorm` 不同。教材的小数据训练示例不支持“Transformer 必然胜过 CNN”的结论。

### 11.9 大规模预训练 Transformer

[原文](../../chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.md)。按编码器型 BERT、编码器—解码器型 T5、解码器型 GPT 区分可见上下文、训练目标和下游接口。微调（fine-tuning）会更新参数，上下文中的少样本提示通常不更新参数。规模规律（scaling laws）描述特定实验条件下模型、数据与计算投入的经验关系，不是无限扩大规模必然解决所有问题的定理。本节作为教材所述模型与研究脉络阅读，不作为当前模型产品或能力排名。

## 代码阅读路线

先用 [评分函数的 `masked_softmax`、`DotProductAttention`](../../chapter_attention-mechanisms-and-transformers/attention-scoring-functions.md) 手算两个查询与三个键的权重矩阵，再跟踪 [多头注意力的 `transpose_qkv`、`transpose_output`](../../chapter_attention-mechanisms-and-transformers/multihead-attention.md)。随后阅读 [Transformer 的 `TransformerEncoderBlock`、`TransformerDecoderBlock`](../../chapter_attention-mechanisms-and-transformers/transformer.md)，逐项标注 Q/K/V 来源和 mask。最后比较 [ViT 的 `PatchEmbedding`、`ViTBlock`](../../chapter_attention-mechanisms-and-transformers/vision-transformer.md)。

## 易错点与自测

- **源序列长 7、目标序列长 5 时，交叉注意力权重形状是什么？** 每个样本、每个头为 $5\times7$；行对应目标查询，列对应源键。
- **Transformer 能并行训练，为何生成仍逐 token？** 自回归目标在推断时依赖尚未产生的历史输出，训练时真实目标前缀已知。
- **进入评估模式是否自动保证因果性？** 不是。教材逐 token 推断只把截至当前的缓存交给注意力；若改为一次输入完整目标序列，仍必须检查因果 mask。
- **位置编码与 token 嵌入相加为什么要求同维度？** 同位置相加要求表示空间尺寸一致；拼接会改变后续投影的输入尺寸。

## 小实验建议（本次未执行）

1. 构造 4 个 token 的小样本，将最后一个 token 改动，比较启用与关闭因果 mask 时前面位置的输出。关闭 dropout 并固定参数，检验未来信息是否泄漏；不需要训练大型模型。
2. 固定图像尺寸和隐藏维度，改变 ViT 的 patch 大小，记录 token 数、注意力矩阵元素数与实测显存。理论增长可预测，实际耗时仍受内核和批量影响；本次没有生成测量结果。
