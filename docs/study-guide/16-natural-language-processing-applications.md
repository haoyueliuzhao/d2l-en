# 第 16 章 自然语言处理：应用重点解析

[返回解析目录](README.md) · [原章目录](../../chapter_natural-language-processing-applications/index.md) · [术语对照表](../GLOSSARY.zh-CN.md)

## 本章定位与先修

本章把预训练表示用于两个任务：单段文本的情感分析（sentiment analysis），以及文本对的自然语言推断（natural language inference, NLI）。需要先理解词嵌入、双向循环网络、一维卷积、注意力与 BERT。本章真正要学的是“输入如何组织、监督目标是什么、如何把变长文本变成预测”，不是把所有任务统一套进一个模型。

学习时比较两种迁移方式：加载 GloVe 后冻结或更新嵌入；加载 BERT 后更新上下文编码器与新分类头。它们的可训练参数量、训练代价和适用输入不同，不能仅凭“用了预训练”就认为实验条件相同。

## 核心机制与公式

句级分类需要把长度为 $T$ 的隐藏状态变成固定维度表示。本章 BiRNN 实现将双向 LSTM 的首末时间步输出拼接；每个时间步有 $2h$ 个特征，因此分类器输入是 $4h$，不是 $2h$。textCNN 对不同窗口大小分别卷积并沿时间做最大池化（max-over-time pooling），突出整段中最强的局部模式。

可分解注意力（decomposable attention）对前提词向量 $\mathbf a_i$ 与假设词向量 $\mathbf b_j$ 计算：

$$
e_{ij}=F(\mathbf a_i)^\top F(\mathbf b_j),\qquad
\boldsymbol\beta_i=\sum_j\operatorname{softmax}_{j}(e_{ij})\mathbf b_j.
$$

$F$ 是共享映射，$\boldsymbol\beta_i$ 是与前提位置 $i$ 对齐的假设表示；反方向也要单独归一化。模型再比较每个词与对齐表示，汇总得到句对标签。BERT 分类则以 `<cls>` 的编码结果经过分类头输出 logits，通过交叉熵训练；不是重新训练 MLM/NSP 两个预训练头。

## 按教材顺序逐节解析

### 16.1 情感分析与数据集

[原文与代码](../../chapter_natural-language-processing-applications/sentiment-analysis-and-dataset.md)。IMDb 的正负评论构成二分类数据，读取文件后分词、建词表，再截断或填充成固定长度。词表应由训练数据确定，测试样本沿用同一映射；长评论截断会丢信息，长度是任务设计的一部分。这里预测的是评论整体倾向，不要求每个词都表达相同情绪。

### 16.2 使用循环神经网络的情感分析

[原文与代码](../../chapter_natural-language-processing-applications/sentiment-analysis-rnn.md)。`BiRNN` 先加载预训练词向量，再以双向 LSTM 编码评论。原实现拼接首末时间步输出，并冻结嵌入参数；应检查 `inputs.T` 后时间维与批量维的变化。固定长度 padding 会影响“最后一步”的含义，教材简化实现不能直接等同于按真实长度打包的变长序列实现。

### 16.3 使用卷积神经网络的情感分析

[原文与代码](../../chapter_natural-language-processing-applications/sentiment-analysis-cnn.md)。一维卷积沿词序扫描，卷积核跨越完整的输入通道；多个窗口大小提取不同长度的局部模式。textCNN 同时使用可训练和冻结的预训练嵌入，将通道拼接后卷积、池化、分类。最大池化保留“某种模式是否强烈出现”，也会弱化其精确位置和多次出现的数量。

### 16.4 自然语言推断与数据集

[原文与代码](../../chapter_natural-language-processing-applications/natural-language-inference-and-dataset.md)。SNLI 输入是前提（premise）与假设（hypothesis），标签是蕴含（entailment）、矛盾（contradiction）、中立（neutral）。中立表示前提不足以支持或否定假设，不等于两段文本毫无相关性。数据加载过滤无有效标签的记录，两个句子分别转为索引并沿用训练词表。

### 16.5 使用注意力的自然语言推断

[原文与代码](../../chapter_natural-language-processing-applications/natural-language-inference-attention.md)。模型分为对齐、比较、聚合三个阶段：先找到跨句相关位置，再比较对应信息，最后对各位置结果求和后分类。它说明无需先把两个句子各自压成一个向量，也能建模句对关系。阅读 `Attend` 时特别核对两个 softmax 的方向；原示例没有完整演示所有 padding 掩码处理，注意力分数不自动等于可解释的语言学关系。

### 16.6 面向序列级与词元级应用微调 BERT

[原文与代码](../../chapter_natural-language-processing-applications/finetuning-bert.md)。单文本分类和句对分类通常读取 `<cls>`；词元标注（text tagging）在每个词元位置预测；抽取式问答（extractive question answering）预测答案起止位置。任务头随监督粒度改变，编码器可复用。本节问答是从给定文本中抽取片段，不能将其描述为开放式生成问答。

### 16.7 微调 BERT 进行自然语言推断

[原文与代码](../../chapter_natural-language-processing-applications/natural-language-inference-bert.md)。加载与预训练词表匹配的模型权重，把前提和假设拼成 `<cls> A <sep> B <sep>`，截断时为三个特殊词元预留位置。`SNLIBERTDataset` 返回词元、片段和有效长度；`BERTClassifier` 复用编码器与隐藏层，新建三分类输出层。加载权重后仍要训练下游分类任务，预训练损失较低不直接给出 NLI 准确率。

## 代码阅读路线

情感任务先读 `load_data_imdb`，然后并排看 `BiRNN.forward` 与 `TextCNN.forward`，画出“词 ID → 嵌入 → 特征聚合 → 两类 logits”的形状变化。句对任务先读 `load_data_snli`，再沿 `Attend → Compare → Aggregate` 阅读；最后对照 `SNLIBERTDataset → BERTClassifier`，确认两条路线使用不同词表及输入格式，不能混用索引。

## 易错点与自测

- **为什么不能用测试集建立新词表？** 答案线索：词 ID 含义必须与训练参数一致；测试集参与特征选择还会改变评估条件。
- **前提与假设互换，标签是否不变？** 答案线索：蕴含有方向性；“动物”与“狗”可构成反例。
- **BERT 微调时哪些参数从零开始？** 答案线索：本例新三分类层从零训练，复用的编码器及隐藏层继续更新；预训练任务头不参与该分类计算。
- **注意力模型为何要做双向对齐？** 答案线索：每个前提词寻找假设对应项，与每个假设词寻找前提对应项，是两个不同的归一化问题。

训练、验证与最终测试必须分工明确；教材示例直接显示测试指标，并不授权在自己的实验中反复用测试集选择窗口、学习率或训练轮数。

## 建议实验（未执行）

1. 保持 IMDb 训练划分、截断长度和训练预算一致，比较 textCNN 冻结全部嵌入与只冻结一套嵌入；记录验证准确率、参数量、耗时和长短评论的错误例子。不要预设解冻一定更优。
2. 为 SNLI 构造少量明确的蕴含、矛盾、中立句对，并加入前提/假设交换版本；分别观察预测与标签变化。人工例子可帮助定位错误，但不能代替完整独立测试集。
